'use client';

import React, { useEffect, useState } from 'react';
import { Plus, MessageSquare, LogOut, Trash2 } from 'lucide-react';

interface SidebarProps {
  token: string;
  activeSessionId: string | null;
  setActiveSessionId: (id: string | null) => void;
  onLogout: () => void;
  refreshTrigger: number;
}

interface SessionSummary {
  id: string;
  title: string | null;
}

export default function Sidebar({ token, activeSessionId, setActiveSessionId, onLogout, refreshTrigger }: SidebarProps) {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchSessions = async () => {
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/sessions`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      if (res.status === 401) {
        onLogout();
        return;
      }
      if (res.ok) {
        const data = await res.json();
        setSessions(data);
      }
    } catch (err) {
      console.error('Failed to fetch sessions:', err);
    }
  };

  useEffect(() => {
    if (token) {
      fetchSessions();
    }
  }, [token, refreshTrigger]);

  const handleCreateSession = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/sessions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ title: 'New Conversation' }),
      });
      if (res.status === 401) {
        onLogout();
        return;
      }
      if (res.ok) {
        const session = await res.json();
        await fetchSessions();
        setActiveSessionId(session.session_id);
      }
    } catch (err) {
      console.error('Failed to create session:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteSession = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (!confirm(`Are you sure you want to delete session "${id}"?`)) return;

    const previousSessions = [...sessions];
    setSessions(prev => prev.filter(session => session.id !== id));
    if (activeSessionId === id) {
      setActiveSessionId(null);
    }

    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/sessions/${id}`, {
        method: 'DELETE',
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      if (res.status === 401) {
        onLogout();
        return;
      }
      if (!res.ok) {
        const errData = await res.json();
        alert(errData.detail || 'Failed to delete session');
        setSessions(previousSessions);
        if (activeSessionId === id) {
          setActiveSessionId(id);
        }
      }
    } catch (err) {
      console.error('Failed to delete session:', err);
      setSessions(previousSessions);
      if (activeSessionId === id) {
        setActiveSessionId(id);
      }
    }
  };

  return (
    <div className="w-64 bg-slate-900 text-slate-100 flex flex-col h-full border-r border-slate-800">
      <div className="p-4 border-b border-slate-800 flex items-center justify-between">
        <h1 className="font-bold text-lg tracking-wider text-slate-200">RAG Assistant</h1>
      </div>

      <div className="p-4">
        <button
          onClick={handleCreateSession}
          disabled={loading}
          className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg py-2.5 px-4 transition-colors font-medium text-sm shadow-md cursor-pointer"
        >
          <Plus size={16} />
          {loading ? 'Creating...' : 'New Conversation'}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-2 space-y-1 scrollbar-thin scrollbar-thumb-slate-800 scrollbar-track-transparent">
        <span className="text-xs font-semibold uppercase text-slate-500 px-2 block mb-2">Conversations</span>
        {sessions.map((session) => (
          <div
            key={session.id}
            onClick={() => setActiveSessionId(session.id)}
            className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-sm text-left transition-colors font-medium cursor-pointer group ${
              activeSessionId === session.id
                ? 'bg-slate-800 text-indigo-400 border border-slate-700'
                : 'hover:bg-slate-800/50 text-slate-400 hover:text-slate-200'
            }`}
          >
            <div className="flex items-center gap-3 truncate">
              <MessageSquare size={16} className={activeSessionId === session.id ? 'text-indigo-400' : 'text-slate-500'} />
              <span className="truncate">{session.title || 'New Conversation'}</span>
            </div>
            <button
              onClick={(e) => handleDeleteSession(e, session.id)}
              className="text-slate-500 hover:text-red-400 p-1 rounded hover:bg-slate-750 transition-colors opacity-0 group-hover:opacity-100 focus:opacity-100 cursor-pointer"
              title="Delete session"
            >
              <Trash2 size={14} />
            </button>
          </div>
        ))}
        {sessions.length === 0 && (
          <p className="text-xs text-slate-500 italic px-2">No active conversations.</p>
        )}
      </div>

      <div className="p-4 border-t border-slate-800">
        <button
          onClick={onLogout}
          className="w-full flex items-center gap-2 text-slate-400 hover:text-red-400 text-sm font-medium px-3 py-2 rounded-lg hover:bg-slate-800/30 transition-colors cursor-pointer"
        >
          <LogOut size={16} />
          Logout
        </button>
      </div>
    </div>
  );
}
