'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { apiError, apiFetch } from '../lib/api';
import { publishCrossTabEvent } from '../lib/broadcast';
import { Document } from '../types';
import { useBroadcastSync } from './useBroadcastSync';

export function useDocuments(token: string | null, onUnauthorized: () => void) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const previousRef = useRef<Document[]>([]);
  const fetchDocuments = useCallback(async () => {
    if (!token) return;
    const response = await apiFetch('/api/v1/documents', token);
    if (response.status === 401) return onUnauthorized();
    if (!response.ok) return;
    const next: Document[] = await response.json();
    const statusChanged = next.some((item) => previousRef.current.some(
      (old) => old.document_id === item.document_id && old.status !== item.status
    ));
    previousRef.current = next;
    setDocuments(next);
    if (statusChanged) publishCrossTabEvent({ type: 'documents_changed' });
  }, [token, onUnauthorized]);
  useBroadcastSync((event) => {
    if (event.type === 'documents_changed' || event.type === 'document_status_changed') void fetchDocuments();
  });
  useEffect(() => {
    void fetchDocuments();
  }, [fetchDocuments]);
  useEffect(() => {
    if (!documents.some((item) => item.status === 'PROCESSING')) return;
    const timer = window.setInterval(() => void fetchDocuments(), 3000);
    return () => window.clearInterval(timer);
  }, [documents, fetchDocuments]);

  const uploadDocument = useCallback(async (file: File) => {
    if (!token) throw new Error('Not authenticated');
    const body = new FormData(); body.append('file', file);
    const response = await apiFetch('/api/v1/documents', token, { method: 'POST', body });
    if (response.status === 401) { onUnauthorized(); throw new Error('Unauthorized'); }
    if (!response.ok) throw await apiError(response, 'Upload failed');
    const result = await response.json();
    await fetchDocuments();
    publishCrossTabEvent({ type: 'documents_changed' });
    return result as { document_id: string };
  }, [token, onUnauthorized, fetchDocuments]);

  const deleteDocument = useCallback(async (documentId: string) => {
    if (!token) return;
    const previous = previousRef.current;
    const optimistic = previous.filter((document) => document.document_id !== documentId);
    previousRef.current = optimistic;
    setDocuments(optimistic);
    try {
      const response = await apiFetch(`/api/v1/documents/${documentId}`, token, { method: 'DELETE' });
      if (response.status === 401) { onUnauthorized(); throw new Error('Unauthorized'); }
      if (!response.ok) throw await apiError(response, 'Delete failed');
      await fetchDocuments();
      publishCrossTabEvent({ type: 'documents_changed' });
    } catch (error) {
      previousRef.current = previous;
      setDocuments(previous);
      throw error;
    }
  }, [token, onUnauthorized, fetchDocuments]);

  const retryDocument = useCallback(async (documentId: string) => {
    if (!token) return;
    const response = await apiFetch(`/api/v1/documents/${documentId}/retry`, token, { method: 'POST' });
    if (response.status === 401) { onUnauthorized(); return; }
    if (!response.ok) throw await apiError(response, 'Retry failed');
    await fetchDocuments();
    publishCrossTabEvent({ type: 'documents_changed' });
  }, [token, onUnauthorized, fetchDocuments]);

  return { documents, fetchDocuments, uploadDocument, deleteDocument, retryDocument };
}
