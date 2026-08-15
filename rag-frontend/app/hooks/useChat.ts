'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { apiError, apiFetch } from '../lib/api';
import { publishCrossTabEvent } from '../lib/broadcast';
import { SseParser } from '../lib/sse';
import { ChatMessage, Source } from '../types';
import { useBroadcastSync } from './useBroadcastSync';

const TERMINAL = new Set(['COMPLETED', 'FAILED', 'CANCELLED']);

export function useChat(token: string | null, sessionId: string | null, onUnauthorized: () => void) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [queryLoading, setQueryLoading] = useState(false);
  const messagesRef = useRef<ChatMessage[]>([]);
  const activeRequestsRef = useRef(new Map<string, { assistantMessageId?: string }>());
  const commit = (next: ChatMessage[]) => { messagesRef.current = next; setMessages(next); };

  const mapServerMessages = (items: any[]): ChatMessage[] => {
    const users = new Map(items.filter((item) => item.role === 'user').map((item) => [item.id, item.client_request_id]));
    return items.map((item) => ({
      message_id: item.id,
      sequence_number: item.sequence_number,
      client_request_id: item.client_request_id || (item.parent_message_id ? users.get(item.parent_message_id) : undefined),
      status: item.status,
      role: item.role,
      text: item.content || '',
      bound_document_ids: (item.selected_document_snapshot?.documents || []).map((entry: any) => entry.document_id),
      sources: item.sources?.documents || [],
    }));
  };

  const reconcile = (canonical: ChatMessage[]) => {
    const merged = [...canonical];
    for (const [requestId, active] of activeRequestsRef.current) {
      const localUser = messagesRef.current.find((item) => item.role === 'user' && item.client_request_id === requestId);
      const localAssistant = messagesRef.current.find((item) => item.role === 'assistant' && item.client_request_id === requestId);
      const assistantIndex = merged.findIndex((item) => item.message_id === active.assistantMessageId || item.client_request_id === requestId && item.role === 'assistant');
      const canonicalAssistant = assistantIndex >= 0 ? merged[assistantIndex] : undefined;
      if (canonicalAssistant && TERMINAL.has(canonicalAssistant.status || '')) {
        activeRequestsRef.current.delete(requestId);
        continue;
      }
      if (canonicalAssistant && localAssistant && (localAssistant.text.length > canonicalAssistant.text.length || canonicalAssistant.status === 'STREAMING')) {
        merged[assistantIndex] = { ...canonicalAssistant, text: localAssistant.text, status: canonicalAssistant.status || localAssistant.status };
      } else if (!canonicalAssistant && localAssistant) {
        merged.push(localAssistant);
      }
      if (!merged.some((item) => item.role === 'user' && item.client_request_id === requestId) && localUser) merged.push(localUser);
    }
    merged.sort((a, b) => (a.sequence_number ?? Number.MAX_SAFE_INTEGER) - (b.sequence_number ?? Number.MAX_SAFE_INTEGER));
    commit(merged);
  };

  const fetchHistory = useCallback(async (id: string) => {
    if (!token) return;
    const response = await apiFetch(`/api/v1/sessions/${id}/messages`, token);
    if (response.status === 401) return onUnauthorized();
    if (!response.ok) return commit([]);
    reconcile(mapServerMessages((await response.json()).items));
  }, [token, onUnauthorized]);

  useEffect(() => {
    if (sessionId) void fetchHistory(sessionId); else commit([]);
  }, [sessionId, fetchHistory]);
  useBroadcastSync((event) => {
    if (event.type === 'session_changed' && event.session_id === sessionId) void fetchHistory(sessionId);
  });

  const updateStreamMessage = (requestId: string, updater: (message: ChatMessage) => ChatMessage) => {
    const identity = activeRequestsRef.current.get(requestId);
    const assistantMessageId = identity?.assistantMessageId;
    commit(messagesRef.current.map((item) => {
      const isTarget = item.role === 'assistant' && (
        (assistantMessageId !== undefined && item.message_id === assistantMessageId) ||
        (assistantMessageId === undefined && item.client_request_id === requestId)
      );
      return isTarget ? updater(item) : item;
    }));
  };

  const sendMessage = useCallback(async (question: string, selectedDocumentIds: string[]) => {
    if (!token || !sessionId || queryLoading) return;
    const clientRequestId = crypto.randomUUID();
    activeRequestsRef.current.set(clientRequestId, {});
    const bound = [...selectedDocumentIds];
    commit([...messagesRef.current,
      { role: 'user', text: question, client_request_id: clientRequestId, status: 'PENDING', bound_document_ids: bound },
      { role: 'assistant', text: '', client_request_id: clientRequestId, status: 'STREAMING', bound_document_ids: bound, sources: [] },
    ]);
    setQueryLoading(true);
    const body = JSON.stringify({ session_id: sessionId, client_request_id: clientRequestId, question, selected_document_ids: bound });
    try {
      let response: Response;
      try { response = await apiFetch('/api/v1/chat/query', token, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body }); }
      catch { response = await apiFetch('/api/v1/chat/query', token, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body }); }
      if (response.status === 401) return onUnauthorized();
      if (!response.ok) throw await apiError(response, 'Query failed');
      const reader = response.body?.getReader();
      const parser = new SseParser();
      const handle = (event: { event: string; data: unknown }) => {
        const data = (event.data || {}) as any;
        if (event.event === 'message_start' && data.message_id) {
          activeRequestsRef.current.set(clientRequestId, { assistantMessageId: data.message_id });
          updateStreamMessage(clientRequestId, (item) => ({ ...item, message_id: data.message_id, status: 'STREAMING' }));
        } else if (event.event === 'token') {
          updateStreamMessage(clientRequestId, (item) => ({ ...item, text: item.text + (data.text || '') }));
        } else if (event.event === 'source') {
          updateStreamMessage(clientRequestId, (item) => ({ ...item, sources: [...(item.sources || []), data as Source] }));
        } else if (event.event === 'message_complete') {
          updateStreamMessage(clientRequestId, (item) => ({ ...item, status: data.status || 'COMPLETED' }));
          activeRequestsRef.current.delete(clientRequestId);
        } else if (event.event === 'cancelled') {
          updateStreamMessage(clientRequestId, (item) => ({ ...item, status: 'CANCELLED' }));
          activeRequestsRef.current.delete(clientRequestId);
        } else if (event.event === 'error') {
          updateStreamMessage(clientRequestId, (item) => ({ ...item, status: 'FAILED', text: data.message || 'Query failed' }));
          activeRequestsRef.current.delete(clientRequestId);
        }
      };
      if (reader) {
        const decoder = new TextDecoder();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          parser.push(decoder.decode(value, { stream: true })).forEach(handle);
        }
        parser.push(decoder.decode()).forEach(handle);
        parser.finish().forEach(handle);
      }
      publishCrossTabEvent({ type: 'session_changed', session_id: sessionId });
    } catch (error) {
      updateStreamMessage(clientRequestId, (item) => ({ ...item, status: 'FAILED', text: error instanceof Error ? error.message : 'Query failed' }));
      activeRequestsRef.current.delete(clientRequestId);
    } finally { setQueryLoading(false); }
  }, [token, sessionId, queryLoading, onUnauthorized]);

  return { messages, queryLoading, sendMessage, fetchHistory };
}
