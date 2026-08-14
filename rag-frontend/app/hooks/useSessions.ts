'use client';

import { useCallback, useEffect, useState } from 'react';
import { apiError, apiFetch } from '../lib/api';
import { publishCrossTabEvent } from '../lib/broadcast';
import { SessionSummary } from '../types';
import { useBroadcastSync } from './useBroadcastSync';

export function useSessions(token: string | null, onUnauthorized: () => void) {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const fetchSessions = useCallback(async () => {
    if (!token) return;
    const response = await apiFetch('/api/v1/sessions', token);
    if (response.status === 401) return onUnauthorized();
    if (response.ok) setSessions(await response.json());
  }, [token, onUnauthorized]);
  useBroadcastSync((event) => {
    if (event.type === 'sessions_changed') void fetchSessions();
  });
  useEffect(() => {
    void fetchSessions();
  }, [fetchSessions]);

  const createSession = useCallback(async () => {
    if (!token) return null;
    setLoading(true);
    try {
      const response = await apiFetch('/api/v1/sessions', token, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'New Conversation' }),
      });
      if (response.status === 401) { onUnauthorized(); return null; }
      if (!response.ok) throw await apiError(response, 'Failed to create session');
      const session = await response.json();
      await fetchSessions();
      publishCrossTabEvent({ type: 'sessions_changed' });
      return session.session_id || session.id;
    } finally { setLoading(false); }
  }, [token, onUnauthorized, fetchSessions]);

  const deleteSession = useCallback(async (id: string) => {
    if (!token) return false;
    const previous = sessions;
    setSessions(previous.filter((session) => session.id !== id));
    try {
      const response = await apiFetch(`/api/v1/sessions/${id}`, token, { method: 'DELETE' });
      if (response.status === 401) { onUnauthorized(); throw new Error('Unauthorized'); }
      if (!response.ok) throw await apiError(response, 'Failed to delete session');
      await fetchSessions();
      publishCrossTabEvent({ type: 'sessions_changed' });
      return true;
    } catch (error) {
      setSessions(previous);
      throw error;
    }
  }, [token, onUnauthorized, fetchSessions, sessions]);

  return { sessions, loading, fetchSessions, createSession, deleteSession };
}
