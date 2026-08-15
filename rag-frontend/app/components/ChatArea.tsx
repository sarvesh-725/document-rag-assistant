'use client';

import React, { useEffect, useRef } from 'react';
import { ChatMessage, Document } from '../types';
import MessageBubble from './MessageBubble';

export default function ChatArea({ messages, documents }: { messages: ChatMessage[]; documents: Document[] }) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  return <div className="flex-1 overflow-y-auto p-6 space-y-6">{messages.map((message, index) => <MessageBubble key={message.message_id || `${message.role}-${message.client_request_id || index}`} message={message} documents={documents} />)}<div ref={endRef} /></div>;
}
