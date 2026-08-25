'use client';

import React from 'react';
import { Source } from '../types';

export default function SourceList({ sources }: { sources?: Source[] }) {
  if (!sources?.length) return null;
  return <div className="flex flex-wrap gap-2 mt-2">{sources.map((source) => {
    const pageLabel = source.page === undefined
      ? ''
      : source.page_end !== undefined && source.page_end !== source.page
        ? `, pp. ${source.page}-${source.page_end}`
        : `, p. ${source.page}`;
    return <div key={source.source_id} className="text-[10px] bg-slate-950 border border-slate-800 rounded-lg px-2 py-1 text-slate-400"><strong className="text-indigo-400">[{source.source_id}]</strong> {source.display_name}{pageLabel}{source.section ? `, ${source.section}` : ''}</div>;
  })}</div>;
}
