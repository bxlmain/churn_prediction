package modelclient

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
)

const inferenceURL = "http://localhost:5000/predict"

type PredictionRequest struct {
	Features map[string]interface{} `json:"features"`
}

type PredictionResponse struct {
	ChurnProbability float64 `json:"churn_probability"`
	ChurnPrediction  int     `json:"churn_prediction"`
}

func GetChurnPrediction(features map[string]interface{}) (*PredictionResponse, error) {
	reqBody := PredictionRequest{Features: features}
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	resp, err := http.Post(inferenceURL, "application/json", bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("inference request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("server returned status: %s", resp.Status)
	}

	var result PredictionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	return &result, nil
}
