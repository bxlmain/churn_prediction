package main

import (
	"churn_backend/features"
	"churn_backend/modelclient"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	_ "github.com/lib/pq"
)

var builder *features.FeatureBuilder

type ChurnRequest struct {
	CustomerID   int    `json:"customer_id"`
	SnapshotDate string `json:"snapshot_date"`
}

type ChurnResponse struct {
	CustomerID       int     `json:"customer_id"`
	SnapshotDate     string  `json:"snapshot_date"`
	ChurnProbability float64 `json:"churn_probability"`
	ChurnPrediction  int     `json:"churn_prediction"`
}

func predictHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ChurnRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid body: %v", err), http.StatusBadRequest)
		return
	}

	snapDate, err := time.Parse("2006-01-02", req.SnapshotDate)
	if err != nil {
		http.Error(w, "Invalid snapshot_date format, use YYYY-MM-DD", http.StatusBadRequest)
		return
	}

	numFeat, catFeat, err := builder.BuildFeatureVector(req.CustomerID, snapDate)
	if err != nil {
		log.Printf("Feature building error: %v", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	allFeatures := make(map[string]interface{})
	for k, v := range numFeat {
		allFeatures[k] = v
	}
	for k, v := range catFeat {
		allFeatures[k] = v
	}

	pred, err := modelclient.GetChurnPrediction(allFeatures)
	if err != nil {
		log.Printf("Model client error: %v", err)
		http.Error(w, "Model unavailable", http.StatusServiceUnavailable)
		return
	}

	response := ChurnResponse{
		CustomerID:       req.CustomerID,
		SnapshotDate:     req.SnapshotDate,
		ChurnProbability: pred.ChurnProbability,
		ChurnPrediction:  pred.ChurnPrediction,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func main() {
	connStr := "host=localhost port=5432 user=churn password=12345 dbname=churn sslmode=disable"
	db, err := sql.Open("postgres", connStr)
	if err != nil {
		log.Fatalf("Cannot open database: %v", err)
	}
	defer db.Close()

	if err = db.Ping(); err != nil {
		log.Fatalf("Cannot ping database: %v", err)
	}

	builder = features.NewFeatureBuilder(db)

	http.HandleFunc("/api/v1/churn", predictHandler)

	port := ":8081" // если занят, поменяйте на другой
	log.Printf("Churn prediction backend starting on %s", port)
	log.Fatal(http.ListenAndServe(port, nil))
}
