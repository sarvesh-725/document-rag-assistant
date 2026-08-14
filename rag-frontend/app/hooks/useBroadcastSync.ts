'use client';

import { useEffect, useRef } from 'react';
import {
  CrossTabEvent,
  publishCrossTabEvent,
  subscribeToCrossTabEvents,
} from '../lib/broadcast';

export function useBroadcastSync(
  onEvent: (event: CrossTabEvent) => void
): (event: CrossTabEvent) => void {
  const listenerRef = useRef(onEvent);
  listenerRef.current = onEvent;

  useEffect(() => subscribeToCrossTabEvents((event) => listenerRef.current(event)), []);
  return publishCrossTabEvent;
}
