import { LogicalReplicationService, PgoutputPlugin } from 'pg-logical-replication';
import { config } from '../config';
import { emitEvent } from './socket';
import { getUserById } from '../features/users/service';

// Tracked tables for CDC
const TRACKED_TABLES = ['users'] as const;
type TrackedTable = (typeof TRACKED_TABLES)[number];

let cdcService: LogicalReplicationService | null = null;
let isConnecting = false;
let retryCount = 0;
const MAX_RETRIES = 50;

// Exponential backoff: 1s, 2s, 4s, 8s, max 30s
const getRetryDelay = (attempt: number): number => {
    return Math.min(1000 * Math.pow(2, attempt), 30000);
};

const scheduleRetry = (): void => {
    if (retryCount >= MAX_RETRIES) {
        console.error('CDC: Max retries reached, giving up');
        return;
    }

    const delay = getRetryDelay(retryCount);
    retryCount++;
    console.log(`CDC: Scheduling retry ${retryCount}/${MAX_RETRIES} in ${delay}ms`);
    setTimeout(() => startCDCConsumer(true), delay);
};

export const startCDCConsumer = async (isRetry = false): Promise<void> => {
    if (config.NODE_ENV === 'test') {
        return;
    }
    if (isConnecting) {
        return;
    }

    isConnecting = true;

    try {
        if (!isRetry) {
            console.log('CDC: Starting consumer...');
        } else {
            console.log(`CDC: Retrying connection (attempt ${retryCount + 1}/${MAX_RETRIES})...`);
        }

        const service = new LogicalReplicationService(
            {
                host: config.POSTGRES_HOST,
                port: config.POSTGRES_PORT,
                database: config.POSTGRES_DATABASE,
                user: config.POSTGRES_USER,
                password: config.POSTGRES_PASSWORD,
            },
            {
                acknowledge: {
                    auto: true,
                    timeoutSeconds: 10,
                },
            }
        );

        cdcService = service;

        service.on('data', async (_lsn: string, changeLog: any) => {
            try {
                await handleDatabaseChange(changeLog);
            } catch (error) {
                console.error('CDC: Error handling change:', error);
            }
        });

        service.on('error', (err: Error) => {
            console.error('CDC: Service error:', err);
            if (cdcService) {
                cdcService = null;
                retryCount = 0;
                scheduleRetry();
            }
        });

        // Subscribe to the publication
        await service.subscribe(
            new PgoutputPlugin({
                protoVersion: 1,
                publicationNames: ['blokus_ai_cdc'],
            }),
            'blokus_ai_cdc_slot'
        );

        console.log('CDC: Consumer started successfully');
        retryCount = 0;
        isConnecting = false;
    } catch (error: any) {
        isConnecting = false;

        // Check if slot is held by another process
        const isSlotActive =
            error.code === '55006' ||
            (error.message && error.message.includes('is active for PID'));

        if (isSlotActive) {
            console.warn('CDC: Replication slot is held by another process. Will retry...');
        } else {
            console.error('CDC: Failed to start consumer:', error);
        }

        scheduleRetry();
    }
};

export const stopCDCConsumer = async (): Promise<void> => {
    if (cdcService) {
        await cdcService.stop();
        cdcService = null;
        console.log('CDC: Consumer stopped');
    }
};

// Handle database changes
const handleDatabaseChange = async (changeLog: any): Promise<void> => {
    const table = changeLog.relation?.name as TrackedTable;
    if (!table || !TRACKED_TABLES.includes(table)) {
        return;
    }

    const operation = changeLog.tag; // 'insert', 'update', 'delete'

    switch (table) {
        case 'users':
            await handleUserChange(operation, changeLog);
            break;
    }
};

// Handle user table changes
const handleUserChange = async (
    operation: string,
    changeLog: any
): Promise<void> => {
    const userId = changeLog.new?.id || changeLog.old?.id;

    if (operation === 'insert' && userId) {
        const user = await getUserById(userId);
        if (user) {
            emitEvent('user-created', { user });
        }
    } else if (operation === 'update' && userId) {
        const user = await getUserById(userId);
        if (user) {
            emitEvent('user-updated', { user });
        }
    } else if (operation === 'delete' && userId) {
        emitEvent('user-deleted', { userId });
    }
};
