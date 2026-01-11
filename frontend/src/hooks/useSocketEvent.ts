import { useEffect } from 'react';
import { apiSocket } from '../api/config';
import { debug } from '../api/debug';
import type { SocketEvent, EventPayloads } from '@blokus-ai/shared';

/**
 * Hook to listen to WebSocket events with type safety.
 *
 * @example
 * useSocketEvent('user-created', (data) => {
 *   console.log('New user:', data.user);
 *   setUsers(prev => [...prev, data.user]);
 * });
 */
export const useSocketEvent = <T extends SocketEvent>(
    eventName: T,
    callback: (data: EventPayloads[T]) => void,
    deps: React.DependencyList = []
) => {
    useEffect(() => {
        if (!apiSocket) {
            console.warn('Socket not initialized');
            return;
        }

        const handleEvent = (serializedData: string) => {
            try {
                const data = JSON.parse(serializedData) as EventPayloads[T];
                callback(data);
            } catch (error) {
                debug.error('Failed to parse socket event data', { error, eventName });
            }
        };

        apiSocket.on(eventName as string, handleEvent);

        return () => {
            apiSocket?.off(eventName as string, handleEvent);
        };
    }, [eventName, ...deps]);
};
