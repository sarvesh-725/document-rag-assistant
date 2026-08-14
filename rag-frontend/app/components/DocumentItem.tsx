'use client';

import React from 'react';
import { AlertTriangle, CheckCircle, Loader2, Trash2 } from 'lucide-react';
import { Document } from '../types';

export function DocumentItem({ document, selected, onToggle, onDelete, onRetry }: {
  document: Document;
  selected: boolean;
  onToggle: (id: string) => void;
  onDelete: (id: string) => Promise<void>;
  onRetry?: (id: string) => Promise<void>;
}) {
  const processing = document.status === 'PROCESSING';
  const failed = document.status === 'FAILED';
  const disabled = processing || document.status === 'DELETING' || failed;
  return (
    <div className="flex items-center justify-between gap-2 p-2 rounded-lg border border-slate-800/60 bg-slate-950/30">
      <label className={`flex items-center gap-2 min-w-0 ${disabled ? 'opacity-70' : 'cursor-pointer'}`}>
        <input type="checkbox" checked={selected} disabled={disabled} onChange={() => onToggle(document.document_id)} />
        <span className="min-w-0"><span className="block truncate text-xs text-slate-300">{document.display_name}</span><time className="block text-[10px] text-slate-600">{new Date(document.created_at).toLocaleString()}</time></span>
      </label>
      <div className="flex items-center gap-2 shrink-0 text-[10px] font-semibold">
        {processing && <span className="text-indigo-400 flex items-center gap-1"><Loader2 size={11} className="animate-spin" />Ingesting...</span>}
        {document.status === 'READY' && <span className="text-emerald-400 flex items-center gap-1"><CheckCircle size={11} />READY</span>}
        {failed && <span className="text-red-400 flex items-center gap-1"><AlertTriangle size={11} />FAILED</span>}
        {failed && onRetry && <button onClick={() => void onRetry(document.document_id)} className="text-amber-400 hover:text-amber-300">Retry</button>}
        <button onClick={() => void onDelete(document.document_id)} title="Delete document" className="text-slate-500 hover:text-red-400"><Trash2 size={12} /></button>
      </div>
    </div>
  );
}
