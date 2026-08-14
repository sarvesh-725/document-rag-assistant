'use client';

import { useCallback, useEffect, useState } from 'react';
import { apiFetch, apiError } from '../lib/api';
import { publishCrossTabEvent } from '../lib/broadcast';
import { SseParser } from '../lib/sse';
import { ChatMessage, Source } from '../types';
import { useBroadcastSync } from './useBroadcastSync';

export function useChat(token: string | null, sessionId: string | null, onUnauthorized: () => void) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [queryLoading, setQueryLoading] = useState(false);
  const fetchHistory = useCallback(async (id: string) => {
    if (!token) return;
    const response = await apiFetch(`/api/v1/sessions/${id}/history`, token);
    if (response.status === 401) return onUnauthorized();
    if (!response.ok) return setMessages([]);
    const data = await response.json();
    setMessages(data.items.map((message: any) => ({
      role: message.role,
      text: message.content || '',
      bound_document_ids: (message.selected_document_snapshot?.documents || []).map((item: any) => item.document_id),
      sources: message.sources?.documents || [],
    })));
  }, [token, onUnauthorized]);
  useEffect(() => { if (sessionId) void fetchHistory(sessionId); else setMessages([]); }, [sessionId, fetchHistory]);
  useBroadcastSync((event) => {
    if (event.type === 'session_changed' && event.session_id === sessionId) void fetchHistory(sessionId);
  });

  const sendMessage = useCallback(async (question: string, selectedDocumentIds: string[]) => {
    if (!token || !sessionId || queryLoading) return;
    const clientRequestId = crypto.randomUUID();
    const bound = [...selectedDocumentIds];
    setMessages((previous) => [...previous,
      { role: 'user', text: question, bound_document_ids: bound },
      { role: 'assistant', text: '', bound_document_ids: bound, sources: [] },
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
        if (event.event === 'token') setMessages((previous) => previous.map((item, index) => index === previous.length - 1 ? { ...item, text: item.text + (data.text || '') } : item));
        if (event.event === 'source') setMessages((previous) => previous.map((item, index) => index === previous.length - 1 ? { ...item, sources: [...(item.sources || []), data as Source] } : item));
        if (event.event === 'error') setMessages((previous) => previous.map((item, index) => index === previous.length - 1 ? { ...item, text: `Error: ${data.message || 'Query failed'}` } : item));
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
      setMessages((previous) => previous.map((item, index) => index === previous.length - 1 ? { ...item, text: error instanceof Error ? error.message : 'Query failed' } : item));
    } finally { setQueryLoading(false); }
  }, [token, sessionId, queryLoading, onUnauthorized]);

  return { messages, queryLoading, sendMessage, fetchHistory };
}
