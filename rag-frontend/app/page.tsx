'use client';

import React, { useState, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar';
import SelectedDocumentsBar from './components/SelectedDocumentsBar';
import ChatInputDock from './components/ChatInputDock';
import { Document } from './types';
import {
  publishCrossTabEvent,
  subscribeToCrossTabEvents,
} from './lib/crossTabSync';
import { Bot, User as UserIcon, Loader2, KeyRound, AlertTriangle } from 'lucide-react';

interface Message {
  role: 'user' | 'assistant';
  text: string;
  bound_document_ids?: string[];
}

export default function Home() {
  const [token, setToken] = useState<string | null>(null);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isSignup, setIsSignup] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);
  const [authLoading, setAuthLoading] = useState(false);

  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputQuestion, setInputQuestion] = useState('');
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[]>([]);
  const [queryLoading, setQueryLoading] = useState(false);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const documentsRef = useRef<Document[]>([]);

  useEffect(() => {
    const savedToken = localStorage.getItem('token');
    if (savedToken) {
      setToken(savedToken);
    }
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const fetchDocuments = async () => {
    if (!token) return;
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/documents`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.status === 401) {
        handleLogout();
        return;
      }
      if (res.ok) {
        const data: Document[] = await res.json();
        const statusChanged = data.some((document) => {
          const previous = documentsRef.current.find(
            (item) => item.document_id === document.document_id
          );
          return previous && previous.status !== document.status;
        });
        documentsRef.current = data;
        setDocuments(data);
        if (statusChanged) {
          publishCrossTabEvent({ type: 'documents_changed' });
        }
      }
    } catch (err) {
      console.error('Failed to fetch documents:', err);
    }
  };

  useEffect(() => {
    let timer: NodeJS.Timeout;
    const hasProcessing = documents.some((document) => document.status === 'PROCESSING');
    if (hasProcessing) {
      timer = setInterval(() => {
        fetchDocuments();
      }, 3000);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [documents, token]);

  useEffect(() => {
    if (token) {
      fetchDocuments();
    }
  }, [token]);

  useEffect(() => {
    const availableIds = new Set(
      documents
        .filter((document) => document.status !== 'DELETING')
        .map((document) => document.document_id)
    );
    setSelectedDocumentIds((previous) =>
      previous.filter((documentId) => availableIds.has(documentId))
    );
  }, [documents]);

  const fetchSessionHistory = async (sessionId: string) => {
    if (!token) return;
    try {
      const historyRes = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/sessions/${sessionId}/history`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      if (historyRes.status === 401) {
        handleLogout();
        return;
      }
      if (!historyRes.ok) {
        setMessages([]);
        return;
      }
      const data = await historyRes.json();
      const mapped = data.items.map((msg: any) => {
        const bound_document_ids = (msg.selected_document_snapshot?.documents || [])
          .map((document: { document_id: string }) => document.document_id);
        return {
          role: msg.role,
          text: msg.content || '',
          bound_document_ids,
        };
      });
      setMessages(mapped);
    } catch (err) {
      console.error('Error fetching session history:', err);
      setMessages([]);
    }
  };

  useEffect(() => {
    if (activeSessionId) {
      fetchSessionHistory(activeSessionId);
    }
  }, [activeSessionId, token]);

  useEffect(() => {
    if (!token) return;
    return subscribeToCrossTabEvents((event) => {
      if (
        event.type === 'documents_changed' ||
        event.type === 'document_status_changed'
      ) {
        void fetchDocuments();
      }
      if (
        event.type === 'session_changed' &&
        event.session_id === activeSessionId
      ) {
        void fetchSessionHistory(event.session_id);
      }
    });
  }, [token, activeSessionId]);


  const handleDeleteDocument = async (document_id: string) => {
    if (!token) return;

    const previousDocuments = [...documents];
    const previousSelected = [...selectedDocumentIds];
    setDocuments(prev => prev.filter(document => document.document_id !== document_id));
    setSelectedDocumentIds(prev => prev.filter(id => id !== document_id));

    try {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/documents/${document_id}`,
        {
          method: 'DELETE',
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      if (res.status === 401) {
        handleLogout();
        return;
      }
      if (!res.ok) {
        const errData = await res.json();
        alert(errData.detail || 'Delete failed');
        setDocuments(previousDocuments);
        setSelectedDocumentIds(previousSelected);
      } else {
        await fetchDocuments();
        publishCrossTabEvent({ type: 'documents_changed' });
      }
    } catch (err) {
      console.error('Failed to delete file:', err);
      setDocuments(previousDocuments);
      setSelectedDocumentIds(previousSelected);
    }
  };


  const handleAuthSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError(null);
    setAuthLoading(true);

    const path = isSignup ? '/api/v1/auth/signup' : '/api/v1/auth/login';
    try {
      let body: any;
      let headers: any = {};

      if (isSignup) {
        body = JSON.stringify({ username, password });
        headers['Content-Type'] = 'application/json';
      } else {
        const params = new URLSearchParams();
        params.append('username', username);
        params.append('password', password);
        body = params;
        headers['Content-Type'] = 'application/x-www-form-urlencoded';
      }

      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}${path}`, {
        method: 'POST',
        headers,
        body,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || 'Authentication failed');
      }

      if (isSignup) {
        setIsSignup(false);
        setAuthError('Signup successful! Please log in.');
      } else {
        const jwtToken = data.access_token;
        localStorage.setItem('token', jwtToken);
        setToken(jwtToken);
        setRefreshTrigger((prev) => prev + 1);
      }
    } catch (err: any) {
      setAuthError(err.message || 'An error occurred during authentication');
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    setToken(null);
    setActiveSessionId(null);
    setMessages([]);
    setDocuments([]);
    setSelectedDocumentIds([]);
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputQuestion.trim() || !activeSessionId || !token || queryLoading) return;

    const userQuestion = inputQuestion.trim();
    setInputQuestion('');
    setQueryLoading(true);

    const activeDocumentIds = selectedDocumentIds;

    setMessages((prev) => [
      ...prev,
      { role: 'user', text: userQuestion, bound_document_ids: activeDocumentIds },
      { role: 'assistant', text: '', bound_document_ids: activeDocumentIds }
    ]);

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/chat/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          session_id: activeSessionId,
          client_request_id: crypto.randomUUID(),
          question: userQuestion,
          selected_document_ids: activeDocumentIds,
        }),
      });

      if (response.status === 401) {
        handleLogout();
        return;
      }

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to stream query');
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            const cleanLine = line.trim();
            if (cleanLine.startsWith('data: ')) {
              const jsonStr = cleanLine.substring(6);
              try {
                const parsed = JSON.parse(jsonStr);
                if (parsed.text) {
                  setMessages((prev) => {
                    const updated = [...prev];
                    const lastIdx = updated.length - 1;
                    if (lastIdx >= 0 && updated[lastIdx].role === 'assistant') {
                      updated[lastIdx] = {
                        ...updated[lastIdx],
                        text: updated[lastIdx].text + parsed.text
                      };
                    }
                    return updated;
                  });
                } else if (parsed.error) {
                  setMessages((prev) => {
                    const updated = [...prev];
                    const lastIdx = updated.length - 1;
                    if (lastIdx >= 0 && updated[lastIdx].role === 'assistant') {
                      updated[lastIdx] = {
                        ...updated[lastIdx],
                        text: `Error: ${parsed.error}`
                      };
                    }
                    return updated;
                  });
                }
              } catch (err) {
              }
            }
          }
        }
      }
      await fetchDocuments();
      publishCrossTabEvent({ type: 'session_changed', session_id: activeSessionId });
    } catch (err: any) {
      publishCrossTabEvent({ type: 'session_changed', session_id: activeSessionId });
      setMessages((prev) => {
        const updated = [...prev];
        const lastIdx = updated.length - 1;
        if (lastIdx >= 0 && updated[lastIdx].role === 'assistant') {
          updated[lastIdx] = {
            ...updated[lastIdx],
            text: `Error generating response: ${err.message || 'Server timeout'}`
          };
        }
        return updated;
      });
    } finally {
      setQueryLoading(false);
    }
  };

  if (!token) {
    return (
      <main className="min-h-screen bg-slate-950 flex flex-col items-center justify-center p-6 text-slate-100 antialiased font-sans">
        <div className="w-full max-w-md bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl overflow-hidden p-8 space-y-6">
          <div className="flex flex-col items-center text-center space-y-2">
            <div className="bg-indigo-600/10 p-3 rounded-2xl border border-indigo-500/20 text-indigo-400">
              <KeyRound size={28} />
            </div>
            <h2 className="text-xl font-bold text-slate-100 tracking-tight">
              {isSignup ? 'Create RAG Account' : 'Sign In to Workspace'}
            </h2>
            <p className="text-xs text-slate-400">
              {isSignup ? 'Register user credentials to start building conversation sessions.' : 'Enter credentials to authorize secure layer.'}
            </p>
          </div>

          <form onSubmit={handleAuthSubmit} className="space-y-4">
            <div className="space-y-1">
              <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">Username</label>
              <input
                type="text"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="developer_admin"
                className="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-sm focus:border-indigo-500 focus:outline-none transition-colors"
              />
            </div>

            <div className="space-y-1">
              <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">Password</label>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-sm focus:border-indigo-500 focus:outline-none transition-colors"
              />
            </div>

            {authError && (
              <div className="flex items-start gap-2.5 bg-red-950/20 border border-red-500/20 p-3 rounded-xl text-xs text-red-400 leading-relaxed">
                <AlertTriangle size={16} className="shrink-0 mt-0.5" />
                <span>{authError}</span>
              </div>
            )}

            <button
              type="submit"
              disabled={authLoading}
              className="w-full flex items-center justify-center bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-700 text-white rounded-xl py-3.5 transition-colors font-semibold text-sm shadow-md cursor-pointer"
            >
              {authLoading ? (
                <Loader2 size={16} className="animate-spin" />
              ) : isSignup ? (
                'Create Account'
              ) : (
                'Authenticate Securely'
              )}
            </button>
          </form>

          <div className="text-center">
            <button
              onClick={() => {
                setIsSignup(!isSignup);
                setAuthError(null);
              }}
              className="text-xs text-indigo-400 hover:text-indigo-300 font-medium transition-colors cursor-pointer"
            >
              {isSignup ? 'Already registered? Sign In' : 'Need a new tenant? Sign Up'}
            </button>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-slate-950 flex text-slate-100 antialiased font-sans h-screen overflow-hidden">
      {}
      <Sidebar
        token={token}
        activeSessionId={activeSessionId}
        setActiveSessionId={setActiveSessionId}
        onLogout={handleLogout}
        refreshTrigger={refreshTrigger}
      />

      {}
      <div className="flex-1 flex flex-col h-full bg-slate-950 min-w-0">
        {activeSessionId ? (
          <div className="flex-1 flex flex-col h-full overflow-hidden">
            {}
            <div className="p-4 border-b border-slate-900 flex items-center justify-between shrink-0 bg-slate-950/80 backdrop-blur-md">
              <div>
                <h3 className="font-semibold text-slate-200">{activeSessionId}</h3>
                <p className="text-[10px] text-slate-500">Streaming Generation Protocol Active</p>
              </div>
            </div>

            {}
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              {messages.map((msg, index) => (
                <div
                  key={index}
                  className={`flex flex-col gap-1 ${msg.role === 'user' ? 'items-end' : 'items-start'}`}
                >
                  {}
                  {msg.role === 'user' && msg.bound_document_ids && msg.bound_document_ids.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 mb-1 max-w-2xl">
                      {msg.bound_document_ids.map((documentId) => {
                        const document = documents.find(item => item.document_id === documentId);
                        const displayName = document ? document.display_name : documentId;
                        return (
                          <span
                            key={documentId}
                            className="bg-slate-900 border border-slate-800 text-slate-400 px-2.5 py-1 rounded-full text-[10px] flex items-center gap-1 font-semibold hover:text-slate-300 transition-colors"
                            title={documentId}
                          >
                            📄 {displayName}
                          </span>
                        );
                      })}
                    </div>
                  )}

                  <div className={`flex gap-4 w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    {msg.role !== 'user' && (
                      <div className="bg-indigo-600/10 border border-indigo-500/20 text-indigo-400 p-2.5 rounded-2xl h-10 w-10 flex items-center justify-center shrink-0 shadow-sm">
                        <Bot size={20} />
                      </div>
                    )}

                    <div
                      className={`max-w-2xl px-4 py-3 rounded-2xl border text-sm leading-relaxed shadow-sm ${
                        msg.role === 'user'
                          ? 'bg-indigo-600 border-indigo-500/30 text-white rounded-tr-none'
                          : 'bg-slate-900 border-slate-800 text-slate-300 rounded-tl-none'
                      }`}
                    >
                      {msg.text ? (
                        msg.text
                      ) : (
                        <div className="flex items-center gap-1.5 text-slate-500">
                          <Loader2 size={14} className="animate-spin text-indigo-400" />
                          <span className="text-xs">Streaming context...</span>
                        </div>
                      )}
                    </div>

                    {msg.role === 'user' && (
                      <div className="bg-slate-800 border border-slate-700 text-slate-300 p-2.5 rounded-2xl h-10 w-10 flex items-center justify-center shrink-0 shadow-sm">
                        <UserIcon size={20} />
                      </div>
                    )}
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>

            {}
            <div className="p-4 border-t border-slate-900 bg-slate-950 shrink-0 flex flex-col gap-2">
              {(() => {
                 return (
                    <SelectedDocumentsBar
                     documents={documents.filter((document) => selectedDocumentIds.includes(document.document_id))}
                     onUnbind={async (documentId) => setSelectedDocumentIds(prev => prev.filter(id => id !== documentId))}
                     onDelete={handleDeleteDocument}
                   />
                );
              })()}
              <ChatInputDock
                token={token}
                inputQuestion={inputQuestion}
                setInputQuestion={setInputQuestion}
                onSubmit={handleSendMessage}
                queryLoading={queryLoading}
                documents={documents}
                fetchDocuments={fetchDocuments}
                selectedDocumentIds={selectedDocumentIds}
                setSelectedDocumentIds={setSelectedDocumentIds}
                onDelete={handleDeleteDocument}
              />
            </div>
          </div>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-center p-8 bg-slate-950">
            <div className="max-w-md space-y-4">
              <div className="bg-indigo-600/10 border border-indigo-500/20 text-indigo-400 p-4 rounded-3xl h-16 w-16 mx-auto flex items-center justify-center shadow-md">
                <Bot size={32} />
              </div>
              <h3 className="text-lg font-bold text-slate-200">No active workspace</h3>
              <p className="text-sm text-slate-500 leading-relaxed">
                Select an existing conversation thread from the sidebar pane or create a fresh conversation workspace canvas to start querying.
              </p>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
