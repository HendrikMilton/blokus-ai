/**
 * Admin API client for training data management
 */

import { http } from './config';
import {
    mlAdminSamplesPath,
    mlAdminStatsPath,
    mlAdminTrainPath,
    TrainingSample,
    TrainingSampleCreate,
    TrainingSampleUpdate,
    TrainingSampleListResponse,
    AdminStats,
    TrainRequest,
    TrainResponse,
    SampleStatus,
} from '@blokus-ai/shared';

export const ADMIN_API = {
    /**
     * Create a new training sample by uploading an image
     */
    createSample: async (data: TrainingSampleCreate): Promise<TrainingSample> => {
        return http.post<TrainingSample>(mlAdminSamplesPath, data);
    },

    /**
     * List training samples with optional filtering and pagination
     */
    listSamples: async (params?: {
        status?: SampleStatus;
        page?: number;
        page_size?: number;
    }): Promise<TrainingSampleListResponse> => {
        const queryParams = new URLSearchParams();
        if (params?.status) {
            queryParams.set('status', params.status);
        }
        if (params?.page) {
            queryParams.set('page', params.page.toString());
        }
        if (params?.page_size) {
            queryParams.set('page_size', params.page_size.toString());
        }

        const path = queryParams.toString()
            ? `${mlAdminSamplesPath}?${queryParams.toString()}`
            : mlAdminSamplesPath;

        return http.get<TrainingSampleListResponse>(path);
    },

    /**
     * Get a single training sample by ID
     */
    getSample: async (id: string): Promise<TrainingSample> => {
        return http.get<TrainingSample>(`${mlAdminSamplesPath}/${id}`);
    },

    /**
     * Get the image URL for a training sample
     */
    getSampleImageUrl: (id: string): string => {
        return `${mlAdminSamplesPath}/${id}/image`;
    },

    /**
     * Update a training sample
     */
    updateSample: async (id: string, data: TrainingSampleUpdate): Promise<TrainingSample> => {
        return http.patch<TrainingSample>(`${mlAdminSamplesPath}/${id}`, data);
    },

    /**
     * Delete a training sample
     */
    deleteSample: async (id: string): Promise<{ success: boolean }> => {
        return http.delete<{ success: boolean }>(`${mlAdminSamplesPath}/${id}`);
    },

    /**
     * Re-run detection on a sample
     */
    repredictSample: async (id: string): Promise<TrainingSample> => {
        return http.post<TrainingSample>(`${mlAdminSamplesPath}/${id}/predict`);
    },

    /**
     * Get admin statistics
     */
    getStats: async (): Promise<AdminStats> => {
        return http.get<AdminStats>(mlAdminStatsPath);
    },

    /**
     * Train the model with labeled samples
     */
    trainModel: async (params?: TrainRequest): Promise<TrainResponse> => {
        return http.post<TrainResponse>(mlAdminTrainPath, params);
    },
};
