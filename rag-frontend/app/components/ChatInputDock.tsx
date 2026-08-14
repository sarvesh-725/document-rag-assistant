'use client';

import React, { useRef, useState } from 'react';
import { Paperclip, Search, Send, UploadCloud, Loader2 } from 'lucide-react';
import { Document } from '../types';
import DocumentList from './DocumentList';

export default function ChatInputDock({
  inputQuestion, setInputQuestion, onSubmit, queryLoading, documents, selectedDocumentIds,
  onToggleDocument, onUpload, onDelete, onRetry,
}: {
  inputQuestion: string;
  setInputQuestion: (value: string) => void;
  onSubmit: (event: React.FormEvent) => void;
  queryLoading: boolean;
  documents: Document[];
  selectedDocumentIds: string[];
  onToggleDocument: (id: string) => void;
  onUpload: (file: File) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
  onRetry: (id: string) => Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState('');
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const filtered = documents.filter((document) => document.display_name.toLowerCase().includes(search.toLowerCase()));
  const upload = async (file?: File) => {
    if (!file) return;
    setUploading(true);
    try { await onUpload(file); setOpen(false); } finally { setUploading(false); }
  };
  return <div className="relative w-full">
    {open && <div className="absolute bottom-full left-0 mb-3 w-[460px] max-w-[calc(100vw-2rem)] bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl p-4 space-y-3 z-50">
      <div className="flex justify-between"><h4 className="text-xs font-bold uppercase tracking-wider text-indigo-400">Documents</h4><button onClick={() => setOpen(false)} className="text-xs text-slate-400">Close</button></div>
      <div onClick={() => inputRef.current?.click()} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); void upload(event.dataTransfer.files[0]); }} className="border-2 border-dashed border-slate-800 rounded-xl p-4 flex flex-col items-center gap-2 cursor-pointer">
        <input ref={inputRef} type="file" accept=".pdf,.txt" className="hidden" onChange={(event) => void upload(event.target.files?.[0])} />
        {uploading ? <Loader2 size={20} className="animate-spin text-indigo-400" /> : <UploadCloud size={20} className="text-slate-500" />}
        <span className="text-xs text-slate-300">{uploading ? 'Uploading...' : 'Upload PDF or TXT'}</span>
      </div>
      <div className="relative"><Search size={14} className="absolute left-3 top-2.5 text-slate-500" /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search documents" className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-9 pr-3 py-2 text-xs" /></div>
      <DocumentList documents={filtered} selectedDocumentIds={selectedDocumentIds} onToggle={onToggleDocument} onDelete={onDelete} onRetry={onRetry} />
    </div>}
    <form onSubmit={onSubmit} className="flex gap-2 items-end bg-slate-900 border border-slate-800 rounded-2xl p-2">
      <button type="button" onClick={() => setOpen(!open)} className="text-slate-400 p-2"><Paperclip size={18} /></button>
      <textarea disabled={queryLoading} rows={1} value={inputQuestion} onChange={(event) => setInputQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); if (inputQuestion.trim()) onSubmit(event); } }} placeholder="Ask a question..." className="flex-1 bg-transparent resize-none text-sm py-2 outline-none" />
      <button type="submit" disabled={queryLoading || !inputQuestion.trim()} className="bg-indigo-600 disabled:bg-slate-800 text-white p-3 rounded-xl"><Send size={16} /></button>
    </form>
  </div>;
}
