'use client';

import React from 'react';
import { LogOut, MessageSquare, Plus, Trash2 } from 'lucide-react';
import { SessionSummary } from '../types';

type SidebarProps = {
  sessions: SessionSummary[];
  activeSessionId: string | null;
  loading: boolean;
  onCreate: () => Promise<string | null>;
  onDelete: (id: string) => Promise<boolean>;
  onSelect: (id: string) => void;
  onLogout: () => void;
};

export default function Sidebar({ sessions, activeSessionId, loading, onCreate, onDelete, onSelect, onLogout }: SidebarProps) {
  return (
    <aside className="w-64 bg-slate-900 text-slate-100 flex flex-col h-full border-r border-slate-800">
      <div className="p-4 border-b border-slate-800"><h1 className="font-bold text-lg tracking-wider text-slate-200">RAG Assistant</h1></div>
      <div className="p-4">
        <button onClick={() => void onCreate()} disabled={loading} className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-800 text-white rounded-lg py-2.5 px-4 transition-colors font-medium text-sm">
          <Plus size={16} />{loading ? 'Creating...' : 'New Conversation'}
        </button>
      </div>
      <div className="flex-1 overflow-y-auto px-4 py-2 space-y-1">
        <span className="text-xs font-semibold uppercase text-slate-500 px-2 block mb-2">Conversations</span>
        {sessions.map((session) => (
          <div key={session.id} onClick={() => onSelect(session.id)} className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-sm transition-colors font-medium cursor-pointer group ${activeSessionId === session.id ? 'bg-slate-800 text-indigo-400 border border-slate-700' : 'hover:bg-slate-800/50 text-slate-400'}`}>
            <div className="flex items-center gap-3 truncate"><MessageSquare size={16} /><span className="truncate">{session.title || 'New Conversation'}</span></div>
            <button onClick={(event) => { event.stopPropagation(); if (confirm(`Delete session "${session.title || session.id}"?`)) void onDelete(session.id); }} className="text-slate-500 hover:text-red-400 p-1 opacity-0 group-hover:opacity-100" title="Delete session"><Trash2 size={14} /></button>
          </div>
        ))}
        {sessions.length === 0 && <p className="text-xs text-slate-500 italic px-2">No active conversations.</p>}
      </div>
      <div className="p-4 border-t border-slate-800"><button onClick={onLogout} className="w-full flex items-center gap-2 text-slate-400 hover:text-red-400 text-sm px-3 py-2"><LogOut size={16} />Logout</button></div>
    </aside>
  );
}
