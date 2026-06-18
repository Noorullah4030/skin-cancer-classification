import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

// Shared Gemini client setup
// Note we set User-Agent header to 'aistudio-build' for telemetry compliance
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    }
  }
});

const app = express();
const PORT = 3000;

// Maximum payload size for photo uploads
app.use(express.json({ limit: "15mb" }));
app.use(express.urlencoded({ extended: true, limit: "15mb" }));

// Standard safety check: health check status route
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", message: "Classification backend online" });
});

// Image Classification Proxy using Gemini Vision
app.post("/api/classify", async (req, res) => {
  try {
    const { imageBase64, mimeType } = req.body;
    
    if (!imageBase64 || !mimeType) {
      return res.status(400).json({ error: "Missing imageBase64 or mimeType arguments." });
    }

    // Clean base64 block
    const base64Data = imageBase64.replace(/^data:image\/\w+;base64,/, "");

    // Prepare content parts for Gemini Multimodal Model (using recommended gemini-3.5-flash)
    const imagePart = {
      inlineData: {
        mimeType: mimeType,
        data: base64Data,
      },
    };

    const textPart = {
      text: `You are a medical skin lesion classification pipeline proxy representing the outcomes of a TensorFlow Lite Skin Cancer classifier model.
Analyze the uploaded dermatoscopic or high-quality camera photo of skin tissue.

Provide a diagnostic probability prediction distributed across these classes in this exact order:
0: "Basal Cell Carcinoma"
1: "Melanoma"
2: "Normal Skin"
3: "Benign"

Provide a mathematically correct prediction distribution where probabilities range between 0.0 and 1.0 and sum up precisely to 1.0 (mimicking a softmax output). Align the maximum probability with the predicted class.

Provide your feedback in a structured JSON schema:
- "predictedClass": string which must be exactly one of: "Basal Cell Carcinoma", "Melanoma", "Normal Skin", or "Benign"
- "confidence": percentage float (0.0 to 100.0) matching the prediction probability
- "probabilities": array of 4 floats representing probabilities for [Basal Cell Carcinoma, Melanoma, Normal Skin, Benign] in that exact order (must sum to 1.0)
- "clinicalAnalysis": 2-3 sentences explaining the visual patterns leading to this classification (such as color asymmetry, border irregularity, standard lesion metrics, or benign regular indicators). Be clinical and professional.`,
    };

    // Call Gemini API 
    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: { parts: [imagePart, textPart] },
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          required: ["predictedClass", "confidence", "probabilities", "clinicalAnalysis"],
          properties: {
            predictedClass: {
              type: Type.STRING,
              description: "Must be 'Basal Cell Carcinoma', 'Melanoma', 'Normal Skin', or 'Benign'"
            },
            confidence: {
              type: Type.NUMBER,
              description: "Confidence percentage of predicted class (0.0 to 100.0)"
            },
            probabilities: {
              type: Type.ARRAY,
              items: { type: Type.NUMBER },
              description: "Array of exactly 4 floats representing probabilities summing up to 1.0 [Basal, Melanoma, Normal, Benign]"
            },
            clinicalAnalysis: {
              type: Type.STRING,
              description: "Short, professional clinical text explaining why"
            }
          }
        }
      }
    });

    const valStr = response.text?.trim() || "";
    const parsedData = JSON.parse(valStr);
    
    // Safety corrections
    if (!parsedData.probabilities || parsedData.probabilities.length !== 4) {
      parsedData.probabilities = [0.25, 0.25, 0.25, 0.25];
    }
    
    // Normalize probabilities to sum to exactly 1.0
    const sum = parsedData.probabilities.reduce((a: number, b: number) => a + b, 0);
    if (Math.abs(sum - 1.0) > 0.01 && sum > 0) {
      parsedData.probabilities = parsedData.probabilities.map((p: number) => p / sum);
    }

    res.json(parsedData);

  } catch (error: any) {
    console.error("Back-end Gemini classification failed:", error);
    res.status(500).json({ 
      error: "Failed to evaluate skin photo.", 
      details: error.message || String(error) 
    });
  }
});

// Configure Express + Vite middleware context
async function initializeServer() {
  if (process.env.NODE_ENV !== "production") {
    // Development Mode Hook
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    // Production Assets Hook
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server starting up on http://localhost:${PORT}`);
  });
}

initializeServer().catch(err => {
  console.error("Express initialization failed:", err);
});
