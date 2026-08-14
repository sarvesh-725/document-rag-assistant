'use client';

import { useCallback, useEffect, useState } from 'react';
import { API_URL } from '../lib/api';

export function useAuth() {
  const [token, setToken] = useState<string | null>(null);
  const [authError, setAuthError] = useState<string | null>(null);
  const [authLoading, setAuthLoading] = useState(false);

  useEffect(() => setToken(localStorage.getItem('token')), []);

  const logout = useCallback(() => {
    localStorage.removeItem('token');
    setToken(null);
  }, []);

  const authenticate = useCallback(async (username: string, password: string, signup: boolean) => {
    setAuthLoading(true);
    setAuthError(null);
    try {
      const response = await fetch(`${API_URL}${signup ? '/api/v1/auth/signup' : '/api/v1/auth/login'}`, {
        method: 'POST',
        headers: { 'Content-Type': signup ? 'application/json' : 'application/x-www-form-urlencoded' },
        body: signup
          ? JSON.stringify({ username, password })
          : new URLSearchParams({ username, password }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Authentication failed');
      if (signup) return { signedUp: true };
      localStorage.setItem('token', data.access_token);
      setToken(data.access_token);
      return { signedUp: false };
    } catch (error) {
      setAuthError(error instanceof Error ? error.message : 'Authentication failed');
      return null;
    } finally {
      setAuthLoading(false);
    }
  }, []);

  return { token, authError, authLoading, authenticate, logout };
}
