import { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    // 1. Create model_detections table
    await knex.schema.createTable('model_detections', (table) => {
        table.uuid('id').primary().defaultTo(knex.fn.uuid());
        table.string('sample_id', 36).notNullable();
        table.uuid('model_id').notNullable().references('id').inTable('training_runs').onDelete('CASCADE');
        table.boolean('has_board_detected').notNullable();
        table.boolean('is_entire_board_visible').notNullable();
        table.float('confidence').nullable();
        table.float('classifier_confidence').nullable();
        table.float('bbox_x').nullable();
        table.float('bbox_y').nullable();
        table.float('bbox_width').nullable();
        table.float('bbox_height').nullable();
        table.float('processing_time_ms').notNullable().defaultTo(0);
        table.timestamp('created_at').defaultTo(knex.fn.now()).notNullable();
        table.timestamp('updated_at').defaultTo(knex.fn.now()).notNullable();

        table.unique(['sample_id', 'model_id']);
        table.index('sample_id');
        table.index('model_id');
    });

    // 2. Migrate existing AI detection data from board_detections to model_detections
    // Find the most recent completed training run
    const latestRun = await knex('training_runs')
        .where('status', 'completed')
        .orderBy('completed_at', 'desc')
        .first();

    if (latestRun) {
        // Get all board_detections with AI detections (confidence is not null)
        const existingDetections = await knex('board_detections')
            .whereNotNull('confidence')
            .select(
                'sample_id',
                'has_board_detected',
                'is_entire_board_visible',
                'confidence',
                'classifier_confidence',
                'bbox_x',
                'bbox_y',
                'bbox_width',
                'bbox_height',
                'processing_time_ms',
                'created_at',
                'updated_at'
            );

        // Insert into model_detections with the latest model_id
        if (existingDetections.length > 0) {
            await knex('model_detections').insert(
                existingDetections.map(det => ({
                    sample_id: det.sample_id,
                    model_id: latestRun.id,
                    has_board_detected: det.has_board_detected,
                    is_entire_board_visible: det.is_entire_board_visible,
                    confidence: det.confidence,
                    classifier_confidence: det.classifier_confidence,
                    bbox_x: det.bbox_x,
                    bbox_y: det.bbox_y,
                    bbox_width: det.bbox_width,
                    bbox_height: det.bbox_height,
                    processing_time_ms: det.processing_time_ms || 0,
                    created_at: det.created_at,
                    updated_at: det.updated_at,
                }))
            );
        }
    }

    // 3. Drop AI detection columns from board_detections
    await knex.schema.alterTable('board_detections', (table) => {
        table.dropColumn('has_board_detected');
        table.dropColumn('is_entire_board_visible');
        table.dropColumn('confidence');
        table.dropColumn('classifier_confidence');
        table.dropColumn('bbox_x');
        table.dropColumn('bbox_y');
        table.dropColumn('bbox_width');
        table.dropColumn('bbox_height');
        table.dropColumn('processing_time_ms');
    });
};

export const down = async (knex: Knex): Promise<void> => {
    // 1. Add AI columns back to board_detections
    await knex.schema.alterTable('board_detections', (table) => {
        table.boolean('has_board_detected').notNullable().defaultTo(false);
        table.boolean('is_entire_board_visible').notNullable().defaultTo(false);
        table.float('confidence').nullable();
        table.float('classifier_confidence').nullable();
        table.float('bbox_x').nullable();
        table.float('bbox_y').nullable();
        table.float('bbox_width').nullable();
        table.float('bbox_height').nullable();
        table.float('processing_time_ms').notNullable().defaultTo(0);
    });

    // 2. Find the most recent model and migrate data back
    const latestRun = await knex('training_runs')
        .where('status', 'completed')
        .orderBy('completed_at', 'desc')
        .first();

    if (latestRun) {
        // Get model_detections for the latest model
        const modelDetections = await knex('model_detections')
            .where('model_id', latestRun.id)
            .select(
                'sample_id',
                'has_board_detected',
                'is_entire_board_visible',
                'confidence',
                'classifier_confidence',
                'bbox_x',
                'bbox_y',
                'bbox_width',
                'bbox_height',
                'processing_time_ms'
            );

        // Update board_detections with the data
        for (const det of modelDetections) {
            await knex('board_detections')
                .where('sample_id', det.sample_id)
                .update({
                    has_board_detected: det.has_board_detected,
                    is_entire_board_visible: det.is_entire_board_visible,
                    confidence: det.confidence,
                    classifier_confidence: det.classifier_confidence,
                    bbox_x: det.bbox_x,
                    bbox_y: det.bbox_y,
                    bbox_width: det.bbox_width,
                    bbox_height: det.bbox_height,
                    processing_time_ms: det.processing_time_ms || 0,
                });
        }
    }

    // 3. Drop model_detections table
    await knex.schema.dropTableIfExists('model_detections');
};
