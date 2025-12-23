const axios = require('axios');
const { Pool } = require('pg');
const crypto = require('crypto');

class CertificateAnalysisController {
    constructor() {
        this.aiServiceUrl = process.env.AI_SERVICE_URL || 'http://localhost:8000';
        this.retryQueue = new Map();
        this.maxRetries = 24;
        this.retryInterval = 60 * 60 * 1000; // 1 hour
        
        this.dbPool = new Pool({
            connectionString: process.env.DATABASE_URL,
            max: 20,
            idleTimeoutMillis: 30000,
            connectionTimeoutMillis: 2000,
        });
        
        // Start retry processor
        this._startRetryProcessor();
    }

    async analyzeCertificate(req, res) {
        const { document_url, provider_id } = req.body;
        const request_id = crypto.randomUUID();

        try {
            // Input validation
            if (!document_url || !provider_id) {
                return res.status(400).json({
                    error: 'Missing required fields: document_url and provider_id'
                });
            }

            // Validate URL format
            try {
                new URL(document_url);
            } catch (e) {
                return res.status(400).json({
                    error: 'Invalid document URL format'
                });
            }

            // Check for existing analysis
            const existingAnalysis = await this.getPendingAnalysis(provider_id);
            if (existingAnalysis) {
                return res.status(409).json({
                    error: 'Analysis already in progress for this provider',
                    request_id: existingAnalysis.request_id
                });
            }

            // Call AI service
            const analysisResponse = await this.callAIService({
                document_url,
                provider_id,
                request_id
            });

            // Store result
            await this.storeAnalysisResult(provider_id, analysisResponse);

            // Handle notifications
            await this.handleAnalysisNotification(provider_id, analysisResponse);

            return res.json({
                request_id,
                status: 'analysis_initiated',
                message: 'Certificate analysis started successfully',
                timestamp: new Date().toISOString()
            });

        } catch (error) {
            console.error(`Analysis failed for request ${request_id}:`, error);
            
            // Queue for retry
            await this.queueForRetry({ document_url, provider_id, request_id });
            
            return res.status(202).json({
                request_id,
                status: 'queued_for_retry',
                message: 'Analysis queued due to temporary service unavailability',
                timestamp: new Date().toISOString()
            });
        }
    }

    async callAIService(requestData) {
        const timeout = 45000; // 45 seconds timeout

        try {
            const response = await axios.post(
                `${this.aiServiceUrl}/analyze`,
                requestData,
                {
                    timeout,
                    headers: {
                        'X-API-Key': process.env.AI_SERVICE_API_KEY || 'default-secret-key'
                    }
                }
            );

            if (response.data.status === 'failed') {
                throw new Error(response.data.error);
            }

            return response.data;

        } catch (error) {
            if (error.code === 'ECONNREFUSED' || 
                error.response?.status >= 500 ||
                error.code === 'ETIMEDOUT') {
                throw new Error('AI service temporarily unavailable');
            }
            
            if (error.response?.status === 401) {
                throw new Error('AI service authentication failed');
            }
            
            throw error;
        }
    }

    async storeAnalysisResult(providerId, analysisResult) {
        const query = `
            UPDATE providers 
            SET credential_analysis = $1,
                analysis_status = $2,
                analysis_timestamp = NOW(),
                updated_at = NOW()
            WHERE id = $3
            RETURNING id
        `;

        const values = [
            analysisResult.report,
            analysisResult.report?.status || 'failed',
            providerId
        ];

        try {
            const result = await this.dbPool.query(query, values);
            if (result.rowCount === 0) {
                throw new Error('Provider not found');
            }
        } catch (error) {
            console.error('Failed to store analysis result:', error);
            throw error;
        }
    }

    async getPendingAnalysis(providerId) {
        const query = `
            SELECT credential_analysis->>'request_id' as request_id
            FROM providers 
            WHERE id = $1 
            AND analysis_status IN ('Pending Admin Review', 'Auto-Rejected', 'analysis_initiated')
            AND analysis_timestamp > NOW() - INTERVAL '24 hours'
            LIMIT 1
        `;

        const result = await this.dbPool.query(query, [providerId]);
        return result.rows[0] || null;
    }

    async queueForRetry(analysisRequest) {
        const retryKey = `${analysisRequest.provider_id}_${analysisRequest.request_id}`;
        
        this.retryQueue.set(retryKey, {
            ...analysisRequest,
            retryCount: 0,
            nextRetry: Date.now() + this.retryInterval,
            lastError: 'Initial failure'
        });

        console.log(`Queued request ${analysisRequest.request_id} for retry`);
    }

    _startRetryProcessor() {
        setInterval(() => {
            this.processRetryQueue();
        }, this.retryInterval);
    }

    async processRetryQueue() {
        const now = Date.now();
        
        for (const [key, request] of this.retryQueue.entries()) {
            if (request.nextRetry <= now && request.retryCount < this.maxRetries) {
                try {
                    console.log(`Retrying analysis for request ${request.request_id}, attempt ${request.retryCount + 1}`);
                    
                    const analysisResponse = await this.callAIService(request);
                    await this.storeAnalysisResult(request.provider_id, analysisResponse);
                    await this.handleAnalysisNotification(request.provider_id, analysisResponse);
                    
                    this.retryQueue.delete(key);
                    console.log(`Successfully processed retry for ${request.request_id}`);
                    
                } catch (error) {
                    console.error(`Retry failed for ${request.request_id}:`, error.message);
                    
                    // Update retry info
                    request.retryCount++;
                    request.nextRetry = Date.now() + this.retryInterval;
                    request.lastError = error.message;
                    
                    if (request.retryCount >= this.maxRetries) {
                        // Max retries exceeded
                        console.error(`Max retries exceeded for ${request.request_id}`);
                        await this.handleFailedAnalysis(request);
                        this.retryQueue.delete(key);
                    }
                }
            }
        }
    }

    async handleAnalysisNotification(providerId, analysisResult) {
        const notificationData = {
            provider_id: providerId,
            analysis_id: analysisResult.analysis_id,
            status: analysisResult.report?.status || 'failed',
            authenticity_score: analysisResult.report?.authenticity_score || 0,
            timestamp: new Date().toISOString()
        };

        // In production, integrate with your notification service
        console.log('Notification data:', notificationData);

        // Example: Notify admins for review
        if (analysisResult.report?.status === 'Pending Admin Review') {
            await this.notifyAdmins(notificationData);
        }
    }

    async notifyAdmins(notificationData) {
        // Implementation for admin notifications
        console.log('Admin notification:', notificationData);
    }

    async handleFailedAnalysis(failedRequest) {
        const query = `
            UPDATE providers 
            SET analysis_status = 'Analysis Failed',
                analysis_notes = $1,
                updated_at = NOW()
            WHERE id = $2
        `;

        const notes = `Analysis failed after ${failedRequest.retryCount} retries. Last error: ${failedRequest.lastError}`;

        try {
            await this.dbPool.query(query, [notes, failedRequest.provider_id]);
            console.error(`Certificate analysis permanently failed for provider: ${failedRequest.provider_id}`);
        } catch (error) {
            console.error('Failed to update provider status:', error);
        }
    }

    async getAnalysisStatus(req, res) {
        const { request_id } = req.params;

        try {
            const query = `
                SELECT analysis_status, credential_analysis, analysis_timestamp
                FROM providers 
                WHERE credential_analysis->>'request_id' = $1
                LIMIT 1
            `;

            const result = await this.dbPool.query(query, [request_id]);
            
            if (result.rows.length === 0) {
                return res.status(404).json({ 
                    error: 'Analysis not found',
                    request_id 
                });
            }

            const { analysis_status, credential_analysis, analysis_timestamp } = result.rows[0];

            return res.json({
                request_id,
                status: analysis_status,
                report: credential_analysis,
                timestamp: analysis_timestamp
            });

        } catch (error) {
            console.error('Failed to get analysis status:', error);
            return res.status(500).json({ 
                error: 'Internal server error',
                request_id 
            });
        }
    }
}

module.exports = CertificateAnalysisController;