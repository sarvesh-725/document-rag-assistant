'use client';

import React from 'react';
import { Document } from '../types';
import { DocumentItem } from './DocumentItem';

export default function DocumentList({ documents, selectedDocumentIds, onToggle, onDelete, onRetry }: {
  documents: Document[];
  selectedDocumentIds: string[];
  onToggle: (id: string) => void;
  onDelete: (id: string) => Promise<void>;
  onRetry?: (id: string) => Promise<void>;
}) {
  return <div className="space-y-1.5 max-h-[220px] overflow-y-auto">
    {documents.map((document) => <DocumentItem key={document.document_id} document={document} selected={selectedDocumentIds.includes(document.document_id)} onToggle={onToggle} onDelete={onDelete} onRetry={onRetry} />)}
    {documents.length === 0 && <p className="text-xs text-slate-500 italic text-center py-4">No documents found.</p>}
  </div>;
}
