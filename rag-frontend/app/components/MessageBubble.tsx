'use client';

import React from 'react';
import { Bot, User as UserIcon } from 'lucide-react';
import { ChatMessage, Document } from '../types';
import SourceList from './SourceList';

export default function MessageBubble({ message, documents }: { message: ChatMessage; documents: Document[] }) {
  const names = (message.bound_document_ids || []).map((id) => documents.find((document) => document.document_id === id)?.display_name || id);
  return <div className={`flex flex-col gap-1 ${message.role === 'user' ? 'items-end' : 'items-start'}`}>
    {names.length > 0 && <div className="flex flex-wrap gap-1">{names.map((name) => <span key={name} className="text-[10px] bg-slate-900 border border-slate-800 text-slate-400 px-2 py-1 rounded-full">{name}</span>)}</div>}
    <div className={`flex gap-3 max-w-3xl ${message.role === 'user' ? 'flex-row-reverse' : ''}`}>
      <div className="h-9 w-9 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center shrink-0">{message.role === 'user' ? <UserIcon size={17} /> : <Bot size={17} className="text-indigo-400" />}</div>
      <div className={`px-4 py-3 rounded-2xl border text-sm leading-relaxed ${message.role === 'user' ? 'bg-indigo-600 border-indigo-500/30 text-white' : 'bg-slate-900 border-slate-800 text-slate-300'}`}>{message.text || 'Streaming...' }<SourceList sources={message.sources} /></div>
    </div>
  </div>;
}
