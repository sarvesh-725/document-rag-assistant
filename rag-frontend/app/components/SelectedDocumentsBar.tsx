'use client';

import React from 'react';
import { FileText, X, Undo2, Loader2, AlertTriangle } from 'lucide-react';
import { Document } from '../types';

interface SelectedDocumentsBarProps {
  documents: Document[];
  onUnbind: (document_id: string) => Promise<void>;
  onDelete: (document_id: string) => Promise<void>;
}

export default function SelectedDocumentsBar({
  documents,
  onUnbind,
  onDelete
}: SelectedDocumentsBarProps) {
  if (!documents || documents.length === 0) return null;

  return (
    <div className="w-full px-4 py-2 bg-slate-900/60 backdrop-blur-md border-t border-slate-800 flex items-center gap-2 overflow-x-auto scrollbar-thin scrollbar-thumb-slate-800 scrollbar-track-transparent">
      <div className="flex items-center gap-2 pb-1">
        {documents.map((document) => {
          const isProcessing = document.status === 'PROCESSING';
          const isFailed = document.status === 'FAILED';
          return (
            <div
              key={document.document_id}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-medium transition-all duration-200 shrink-0 ${
                isProcessing
                  ? 'bg-indigo-950/40 border-indigo-500/30 text-indigo-300'
                  : isFailed
                  ? 'bg-red-950/40 border-red-500/30 text-red-300'
                  : 'bg-slate-800 border-slate-700 text-slate-200 hover:border-slate-600'
              }`}
            >
              {isProcessing ? (
                <Loader2 size={12} className="animate-spin text-indigo-400" />
              ) : isFailed ? (
                <AlertTriangle size={12} className="text-red-400" />
              ) : (
                <FileText size={12} className="text-indigo-400" />
              )}
              
              <span className="max-w-[120px] truncate">{document.display_name}</span>

              {isProcessing ? (
                <span className="text-[10px] text-indigo-400 font-semibold uppercase animate-pulse">
                  Ingesting
                </span>
              ) : isFailed ? (
                <div className="flex items-center gap-1.5">
                  <span className="text-[9px] text-red-400 font-semibold uppercase">
                    Failed
                  </span>
                  <button
                    onClick={() => onDelete(document.document_id)}
                    className="flex items-center gap-0.5 bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/20 rounded px-1.5 py-0.5 transition-colors cursor-pointer text-[10px]"
                    title="Remove failed file from list"
                  >
                    Remove
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => onUnbind(document.document_id)}
                  className="text-slate-400 hover:text-slate-200 p-0.5 rounded-full hover:bg-slate-700 transition-colors cursor-pointer"
                  title="Unselect file"
                >
                  <X size={12} />
                </button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
