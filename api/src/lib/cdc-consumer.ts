import { LogicalReplicationService, PgoutputPlugin } from 'pg-logical-replication';
import { config } from '../config';
import { emitEvent } from './socket';

// Tracked tables for CDC - add tables here as features are added
const TRACKED_TABLES: string[] = [];

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
    // Skip if no tables to track or in test mode
    if (config.NODE_ENV === 'test' || TRACKED_TABLES.length === 0) {
        console.log('CDC: No tables to track, skipping consumer');
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

// Handle database changes - add handlers here as features are added
const handleDatabaseChange = async (changeLog: any): Promise<void> => {
    const table = changeLog.relation?.name;
    if (!table || !TRACKED_TABLES.includes(table)) {
        return;
    }

    const operation = changeLog.tag; // 'insert', 'update', 'delete'
    const data = changeLog.new || changeLog.old;

    // Emit generic event for now
    emitEvent(`${table}-${operation}`, { table, operation, data });
};
