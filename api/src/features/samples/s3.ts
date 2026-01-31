import { S3Client, PutObjectCommand, ListObjectsV2Command, HeadObjectCommand, GetObjectCommand, DeleteObjectsCommand } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import { config } from '../../config';
import sharp from 'sharp';

const s3Client = new S3Client({ region: config.AWS_REGION });

const BUCKET = config.AWS_S3_BUCKET;
const ORIGINALS_PREFIX = 'originals/';
const THUMBNAILS_PREFIX = 'thumbnails/';

export interface S3Sample {
    id: string;
    key: string;
    lastModified: Date;
    size: number;
}

export const listSamples = async (): Promise<S3Sample[]> => {
    const command = new ListObjectsV2Command({
        Bucket: BUCKET,
        Prefix: ORIGINALS_PREFIX,
    });

    const response = await s3Client.send(command);
    const samples: S3Sample[] = [];

    for (const obj of response.Contents || []) {
        if (!obj.Key || !obj.Key.endsWith('.png')) {
            continue;
        }

        const filename = obj.Key.replace(ORIGINALS_PREFIX, '');
        const id = filename.replace('.png', '');

        samples.push({
            id,
            key: obj.Key,
            lastModified: obj.LastModified || new Date(),
            size: obj.Size || 0,
        });
    }

    // Sort by lastModified descending
    samples.sort((a, b) => b.lastModified.getTime() - a.lastModified.getTime());

    return samples;
};

export const getSample = async (id: string): Promise<S3Sample | null> => {
    const key = `${ORIGINALS_PREFIX}${id}.png`;

    try {
        const command = new HeadObjectCommand({
            Bucket: BUCKET,
            Key: key,
        });

        const response = await s3Client.send(command);

        return {
            id,
            key,
            lastModified: response.LastModified || new Date(),
            size: response.ContentLength || 0,
        };
    } catch {
        return null;
    }
};

export const getUploadPresignedUrl = async (id: string): Promise<{ url: string; key: string }> => {
    const key = `${ORIGINALS_PREFIX}${id}.png`;

    const command = new PutObjectCommand({
        Bucket: BUCKET,
        Key: key,
        ContentType: 'image/png',
    });

    const url = await getSignedUrl(s3Client, command, { expiresIn: 3600 });

    return { url, key };
};

export const getCloudFrontUrl = (id: string): string => `${config.CLOUDFRONT_URL}/${ORIGINALS_PREFIX}${id}.png`;

export const getThumbnailUrl = (id: string): string => `${config.CLOUDFRONT_URL}/${THUMBNAILS_PREFIX}${id}.jpg`;

export const generateThumbnail = async (id: string): Promise<void> => {
    const originalKey = `${ORIGINALS_PREFIX}${id}.png`;
    const thumbnailKey = `${THUMBNAILS_PREFIX}${id}.jpg`;

    // Download original from S3
    const getCommand = new GetObjectCommand({
        Bucket: BUCKET,
        Key: originalKey,
    });

    const response = await s3Client.send(getCommand);
    const originalBuffer = Buffer.from(await response.Body!.transformToByteArray());

    // Apply EXIF rotation, resize preserving aspect ratio, convert to JPEG
    const thumbnailBuffer = await sharp(originalBuffer)
        .rotate() // Auto-rotate based on EXIF orientation
        .resize(1200, 1200, { fit: 'inside', withoutEnlargement: true }) // Max 1200px, preserve aspect ratio
        .jpeg({ quality: 85 })
        .toBuffer();

    // Upload thumbnail
    const putCommand = new PutObjectCommand({
        Bucket: BUCKET,
        Key: thumbnailKey,
        Body: thumbnailBuffer,
        ContentType: 'image/jpeg',
    });

    await s3Client.send(putCommand);
};

export const thumbnailExists = async (id: string): Promise<boolean> => {
    const key = `${THUMBNAILS_PREFIX}${id}.jpg`;

    try {
        const command = new HeadObjectCommand({
            Bucket: BUCKET,
            Key: key,
        });
        await s3Client.send(command);
        return true;
    } catch {
        return false;
    }
};

export const deleteSamples = async (ids: string[]): Promise<number> => {
    if (ids.length === 0) {
        return 0;
    }

    // Build list of objects to delete (original + thumbnail for each ID)
    const objects = ids.flatMap(id => [
        { Key: `${ORIGINALS_PREFIX}${id}.png` },
        { Key: `${THUMBNAILS_PREFIX}${id}.jpg` },
    ]);

    // S3 DeleteObjects can handle max 1000 objects per request
    const batchSize = 1000;
    let deletedCount = 0;

    for (let i = 0; i < objects.length; i += batchSize) {
        const batch = objects.slice(i, i + batchSize);

        const command = new DeleteObjectsCommand({
            Bucket: BUCKET,
            Delete: {
                Objects: batch,
                Quiet: true,
            },
        });

        const response = await s3Client.send(command);
        // Count successful deletions (errors array contains failed deletions)
        deletedCount += batch.length - (response.Errors?.length ?? 0);
    }

    // Return number of samples deleted (not objects, so divide by 2)
    return Math.floor(deletedCount / 2);
};
