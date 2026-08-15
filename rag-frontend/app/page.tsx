'use client';

import React, { useEffect, useRef, useState } from 'react';
import { AlertTriangle, Bot, KeyRound } from 'lucide-react';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import ChatInputDock from './components/ChatInputDock';
import SelectedFilesBar from './components/SelectedFilesBar';
import { useAuth } from './hooks/useAuth';
import { useChat } from './hooks/useChat';
import { useDocuments } from './hooks/useDocuments';
import { useSessions } from './hooks/useSessions';
import { useBroadcastSync } from './hooks/useBroadcastSync';

export default function Home() {
  const { token, authError, authLoading, authenticate, logout } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [signup, setSignup] = useState(false);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[]>([]);
  const [question, setQuestion] = useState('');
  const selectionSessionRef = useRef<string | null>(null);
  const remoteSelectionRef = useRef(false);
  const selectionKey = (sessionId: string) => `document-rag:selected:${sessionId}`;
  const readSelection = (sessionId: string) => {
    try {
      const value = JSON.parse(localStorage.getItem(selectionKey(sessionId)) || '[]');
      return Array.isArray(value) ? value.filter((id): id is string => typeof id === 'string') : [];
    } catch { return []; }
  };
  const sessions = useSessions(token, logout);
  const documents = useDocuments(token, logout);
  const chat = useChat(token, activeSessionId, logout);
  const publishSelection = useBroadcastSync((event) => {
    if (event.type === 'selection_changed' && event.session_id === activeSessionId) {
      remoteSelectionRef.current = true;
      setSelectedDocumentIds(readSelection(event.session_id));
    }
  });
  useEffect(() => {
    selectionSessionRef.current = activeSessionId;
    setSelectedDocumentIds(activeSessionId ? readSelection(activeSessionId) : []);
  }, [activeSessionId]);
  useEffect(() => {
    if (!activeSessionId || selectionSessionRef.current !== activeSessionId) return;
    if (remoteSelectionRef.current) {
      remoteSelectionRef.current = false;
      return;
    }
    localStorage.setItem(selectionKey(activeSessionId), JSON.stringify(selectedDocumentIds));
    publishSelection({ type: 'selection_changed', session_id: activeSessionId });
  }, [activeSessionId, selectedDocumentIds, publishSelection]);
  useEffect(() => {
    if (activeSessionId && !sessions.sessions.some((session) => session.id === activeSessionId)) {
      setActiveSessionId(null);
    }
  }, [sessions.sessions, activeSessionId]);

  if (!token) {
    return <main className="min-h-screen bg-slate-950 flex items-center justify-center p-6 text-slate-100"><form onSubmit={async (event) => { event.preventDefault(); const result = await authenticate(username, password, signup); if (result?.signedUp) setSignup(false); }} className="w-full max-w-md bg-slate-900 border border-slate-800 rounded-2xl p-8 space-y-5">
      <div className="text-center"><div className="inline-flex bg-indigo-600/10 p-3 rounded-2xl text-indigo-400"><KeyRound size={28} /></div><h2 className="text-xl font-bold mt-3">{signup ? 'Create RAG Account' : 'Sign In to Workspace'}</h2></div>
      <input required value={username} onChange={(event) => setUsername(event.target.value)} placeholder="Username" className="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-sm" />
      <input required type="password" value={password} onChange={(event) => setPassword(event.target.value)} placeholder="Password" className="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-sm" />
      {authError && <div className="flex gap-2 text-xs text-red-400"><AlertTriangle size={15} />{authError}</div>}
      <button disabled={authLoading} className="w-full bg-indigo-600 disabled:bg-indigo-900 text-white rounded-xl py-3 font-semibold">{authLoading ? 'Working...' : signup ? 'Create Account' : 'Authenticate Securely'}</button>
      <button type="button" onClick={() => setSignup(!signup)} className="w-full text-xs text-indigo-400">{signup ? 'Already registered? Sign In' : 'Need an account? Sign Up'}</button>
    </form></main>;
  }

  const upload = async (file: File) => {
    const result = await documents.uploadDocument(file);
    setSelectedDocumentIds((previous) => previous.includes(result.document_id) ? previous : [...previous, result.document_id]);
  };
  const deleteDocument = async (documentId: string) => {
    const confirmed = confirm('This document will be removed from your document library and will no longer be available to future questions. Existing running queries are allowed to finish. Continue?');
    if (!confirmed) return;
    try {
      await documents.deleteDocument(documentId);
      setSelectedDocumentIds((previous) => previous.filter((id) => id !== documentId));
    } catch (error) {
      alert(error instanceof Error ? error.message : 'Delete failed');
    }
  };
  const toggleDocument = (id: string) => setSelectedDocumentIds((previous) => previous.includes(id) ? previous.filter((item) => item !== id) : [...previous, id]);
  const submitQuestion = (value: string) => {
    if (!activeSessionId) return;
    const latestSelection = readSelection(activeSessionId);
    setQuestion('');
    void chat.sendMessage(value, latestSelection);
  };

  return <main className="min-h-screen bg-slate-950 flex text-slate-100 h-screen overflow-hidden">
    <Sidebar sessions={sessions.sessions} activeSessionId={activeSessionId} loading={sessions.loading} onCreate={async () => { const id = await sessions.createSession(); if (id) setActiveSessionId(id); return id; }} onDelete={async (id) => { try { const result = await sessions.deleteSession(id); if (activeSessionId === id) setActiveSessionId(null); return result; } catch (error) { alert(error instanceof Error ? error.message : 'Session delete failed'); return false; } }} onSelect={setActiveSessionId} onLogout={logout} />
    <div className="flex-1 flex flex-col min-w-0">
      {activeSessionId ? <>
        <header className="p-4 border-b border-slate-900"><h3 className="font-semibold">Conversation</h3><p className="text-[10px] text-slate-500">Global document context</p></header>
        <ChatArea messages={chat.messages} documents={documents.documents} />
        <div className="p-4 border-t border-slate-900 space-y-2">
          <SelectedFilesBar documents={documents.documents.filter((document) => selectedDocumentIds.includes(document.document_id))} onUnbind={async (id) => setSelectedDocumentIds((previous) => previous.filter((item) => item !== id))} onDelete={deleteDocument} />
          <ChatInputDock inputQuestion={question} setInputQuestion={setQuestion} onSubmit={(event) => { event.preventDefault(); const value = question.trim(); if (value) submitQuestion(value); }} documents={documents.documents} selectedDocumentIds={selectedDocumentIds} onToggleDocument={toggleDocument} onUpload={upload} onDelete={deleteDocument} onRetry={documents.retryDocument} />
        </div>
      </> : <div className="flex-1 flex items-center justify-center text-center p-8"><div><Bot size={42} className="mx-auto text-indigo-400 mb-4" /><h3 className="font-bold text-lg">No active workspace</h3><p className="text-sm text-slate-500 mt-2">Create or select a conversation to begin.</p></div></div>}
    </div>
  </main>;
}
