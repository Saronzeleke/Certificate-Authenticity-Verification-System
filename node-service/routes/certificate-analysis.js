const express = require('express');
const router = express.Router();
const CertificateAnalysisController = require('../controllers/certificate-analysis');

const analysisController = new CertificateAnalysisController();

// Analysis endpoints
router.post('/analyze', (req, res) => analysisController.analyzeCertificate(req, res));
router.get('/status/:request_id', (req, res) => analysisController.getAnalysisStatus(req, res));

module.exports = router;