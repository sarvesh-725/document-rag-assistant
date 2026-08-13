export const CROSS_TAB_CHANNEL_NAME = 'document-rag-assistant';

export type CrossTabEvent =
  | { type: 'documents_changed' }
  | { type: 'sessions_changed' }
  | { type: 'session_changed'; session_id: string }
  | { type: 'document_status_changed' };

let channel: BroadcastChannel | null = null;

function getChannel(): BroadcastChannel | null {
  if (typeof window === 'undefined' || typeof BroadcastChannel === 'undefined') {
    return null;
  }
  channel ??= new BroadcastChannel(CROSS_TAB_CHANNEL_NAME);
  return channel;
}

export function publishCrossTabEvent(event: CrossTabEvent): void {
  getChannel()?.postMessage(event);
}

export function subscribeToCrossTabEvents(
  listener: (event: CrossTabEvent) => void
): () => void {
  const currentChannel = getChannel();
  if (!currentChannel) return () => undefined;

  const handleMessage = (message: MessageEvent<CrossTabEvent>) => {
    if (message.data && typeof message.data.type === 'string') {
      listener(message.data);
    }
  };
  currentChannel.addEventListener('message', handleMessage);
  return () => currentChannel.removeEventListener('message', handleMessage);
}
