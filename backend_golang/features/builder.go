package features

import (
	"database/sql"
	"fmt"
	"math"
	"time"
)

type CustomerStatic struct {
	RegistrationDate time.Time
	City             string
	Gender           string
	PreferredPayment string
	Age              int
}

type FeatureBuilder struct {
	DB *sql.DB
}

func NewFeatureBuilder(db *sql.DB) *FeatureBuilder {
	return &FeatureBuilder{DB: db}
}

func (fb *FeatureBuilder) getCustomer(customerID int) (*CustomerStatic, error) {
	row := fb.DB.QueryRow(`
		SELECT registration_date, city, gender, preferred_payment, age
		FROM customers_live
		WHERE customer_id = $1`, customerID)
	var reg time.Time
	var city, gender, payment string
	var age int
	err := row.Scan(&reg, &city, &gender, &payment, &age)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("customer %d not found", customerID)
	}
	if err != nil {
		return nil, fmt.Errorf("customer query: %w", err)
	}
	return &CustomerStatic{
		RegistrationDate: reg,
		City:             city,
		Gender:           gender,
		PreferredPayment: payment,
		Age:              age,
	}, nil
}

func (fb *FeatureBuilder) orderFeatures(customerID int, snapDate time.Time) (map[string]float64, error) {
	f := make(map[string]float64)
	snapStr := snapDate.Format("2006-01-02")

	// общие заказы
	var ordersTotal, ordersAmountTotal, ordersAmountMean, ordersQuantityTotal float64
	err := fb.DB.QueryRow(`
		SELECT COUNT(order_id),
			COALESCE(SUM(amount), 0),
			COALESCE(AVG(amount), 0),
			COALESCE(SUM(quantity), 0)
		FROM orders_live
		WHERE customer_id = $1 AND order_date < $2
	`, customerID, snapStr).Scan(&ordersTotal, &ordersAmountTotal, &ordersAmountMean, &ordersQuantityTotal)
	if err != nil {
		return nil, fmt.Errorf("orders agg: %w", err)
	}
	f["orders_total"] = ordersTotal
	f["orders_amount_total"] = ordersAmountTotal
	f["orders_amount_mean"] = ordersAmountMean
	f["orders_quantity_total"] = ordersQuantityTotal

	// доставленные заказы
	var deliveredOrdersTotal, deliveredAmountTotal, deliveredAmountMean float64
	err = fb.DB.QueryRow(`
		SELECT COUNT(order_id),
			COALESCE(SUM(amount), 0),
			COALESCE(AVG(amount), 0)
		FROM orders_live
		WHERE customer_id = $1 AND order_date < $2 AND status = 'delivered'
	`, customerID, snapStr).Scan(&deliveredOrdersTotal, &deliveredAmountTotal, &deliveredAmountMean)
	if err != nil {
		return nil, fmt.Errorf("delivered agg: %w", err)
	}
	f["delivered_orders_total"] = deliveredOrdersTotal
	f["delivered_amount_total"] = deliveredAmountTotal
	f["delivered_amount_mean"] = deliveredAmountMean

	// отменённые
	var cancelled float64
	err = fb.DB.QueryRow(`
		SELECT COUNT(order_id) FROM orders_live
		WHERE customer_id = $1 AND order_date < $2 AND status IN ('cancelled','returned')
	`, customerID, snapStr).Scan(&cancelled)
	if err != nil {
		return nil, fmt.Errorf("cancelled agg: %w", err)
	}
	f["orders_cancelled_total"] = cancelled

	// delivered за 30/90 дней
	for _, days := range []int{30, 90} {
		start := snapDate.AddDate(0, 0, -days).Format("2006-01-02")
		var cnt, amt float64
		err = fb.DB.QueryRow(`
			SELECT COUNT(order_id), COALESCE(SUM(amount), 0)
			FROM orders_live
			WHERE customer_id = $1 AND order_date >= $2 AND order_date < $3 AND status = 'delivered'
		`, customerID, start, snapStr).Scan(&cnt, &amt)
		if err != nil {
			return nil, fmt.Errorf("delivered %dd agg: %w", days, err)
		}
		f[fmt.Sprintf("delivered_orders_%dd", days)] = cnt
		f[fmt.Sprintf("delivered_amount_%dd", days)] = amt
	}

	// days_since_last_order
	var lastOrder sql.NullTime
	err = fb.DB.QueryRow(`
		SELECT MAX(order_date) FROM orders_live
		WHERE customer_id = $1 AND order_date < $2
	`, customerID, snapStr).Scan(&lastOrder)
	if err != nil {
		return nil, err
	}
	if lastOrder.Valid {
		f["days_since_last_order"] = snapDate.Sub(lastOrder.Time).Hours() / 24
	} else {
		f["days_since_last_order"] = -1
	}
	return f, nil
}

func (fb *FeatureBuilder) visitFeatures(customerID int, snapDate time.Time) (map[string]float64, error) {
	f := make(map[string]float64)
	snapStr := snapDate.Format("2006-01-02")

	row := fb.DB.QueryRow(`
		SELECT COUNT(visit_id),
			COALESCE(SUM(pages_viewed), 0),
			COALESCE(AVG(pages_viewed), 0),
			COALESCE(SUM(cart_adds), 0),
			COALESCE(SUM(checkout_initiated), 0),
			COALESCE(AVG(session_duration_sec), 0)
		FROM visits_live
		WHERE customer_id = $1 AND visit_time < $2
	`, customerID, snapStr)
	var visitsTotal, pagesTotal, pagesMean, cartAdds, checkoutInit, sessionMean float64
	if err := row.Scan(&visitsTotal, &pagesTotal, &pagesMean, &cartAdds, &checkoutInit, &sessionMean); err != nil {
		return nil, fmt.Errorf("visits agg: %w", err)
	}
	f["visits_total"] = visitsTotal
	f["pages_viewed_total"] = pagesTotal
	f["pages_viewed_mean"] = pagesMean
	f["cart_adds_total"] = cartAdds
	f["checkout_initiated_total"] = checkoutInit
	f["session_duration_mean"] = sessionMean

	// последние 30/90 дней
	for _, days := range []int{30, 90} {
		start := snapDate.AddDate(0, 0, -days).Format("2006-01-02")
		var vCnt, cAdds, chInit float64
		err := fb.DB.QueryRow(`
			SELECT COUNT(visit_id), COALESCE(SUM(cart_adds), 0), COALESCE(SUM(checkout_initiated), 0)
			FROM visits_live
			WHERE customer_id = $1 AND visit_time >= $2 AND visit_time < $3
		`, customerID, start, snapStr).Scan(&vCnt, &cAdds, &chInit)
		if err != nil {
			return nil, fmt.Errorf("visits %dd agg: %w", days, err)
		}
		f[fmt.Sprintf("visits_%dd", days)] = vCnt
		f[fmt.Sprintf("cart_adds_%dd", days)] = cAdds
		f[fmt.Sprintf("checkout_initiated_%dd", days)] = chInit
	}

	// days_since_last_visit
	var lastVisit sql.NullTime
	err := fb.DB.QueryRow(`
		SELECT MAX(visit_time) FROM visits_live
		WHERE customer_id = $1 AND visit_time < $2
	`, customerID, snapStr).Scan(&lastVisit)
	if err != nil {
		return nil, err
	}
	if lastVisit.Valid {
		f["days_since_last_visit"] = snapDate.Sub(lastVisit.Time).Hours() / 24
	} else {
		f["days_since_last_visit"] = -1
	}
	return f, nil
}

func (fb *FeatureBuilder) supportFeatures(customerID int, snapDate time.Time) (map[string]float64, error) {
	f := make(map[string]float64)
	snapStr := snapDate.Format("2006-01-02")

	var ticketsTotal, ratingMean, resolutionMean sql.NullFloat64
	err := fb.DB.QueryRow(`
		SELECT COUNT(ticket_id),
			AVG(rating),
			AVG(closed_date - created_date)
		FROM support_tickets_live
		WHERE customer_id = $1 AND created_date < $2
	`, customerID, snapStr).Scan(&ticketsTotal, &ratingMean, &resolutionMean)
	if err != nil {
		return nil, fmt.Errorf("support agg: %w", err)
	}
	if ticketsTotal.Valid {
		f["support_tickets_total"] = ticketsTotal.Float64
	} else {
		f["support_tickets_total"] = 0
	}
	if ratingMean.Valid {
		f["support_rating_mean"] = ratingMean.Float64
	} else {
		f["support_rating_mean"] = 0
	}
	if resolutionMean.Valid {
		f["support_resolution_days_mean"] = resolutionMean.Float64
	} else {
		f["support_resolution_days_mean"] = 0
	}

	// последние 90 дней
	start90 := snapDate.AddDate(0, 0, -90).Format("2006-01-02")
	var tickets90 float64
	err = fb.DB.QueryRow(`
		SELECT COUNT(ticket_id) FROM support_tickets_live
		WHERE customer_id = $1 AND created_date >= $2 AND created_date < $3
	`, customerID, start90, snapStr).Scan(&tickets90)
	if err != nil {
		return nil, fmt.Errorf("support 90d agg: %w", err)
	}
	f["support_tickets_90d"] = tickets90

	// days_since_last_ticket
	var lastTicket sql.NullTime
	err = fb.DB.QueryRow(`
		SELECT MAX(created_date) FROM support_tickets_live
		WHERE customer_id = $1 AND created_date < $2
	`, customerID, snapStr).Scan(&lastTicket)
	if err != nil {
		return nil, err
	}
	if lastTicket.Valid {
		f["days_since_last_ticket"] = snapDate.Sub(lastTicket.Time).Hours() / 24
	} else {
		f["days_since_last_ticket"] = -1
	}
	return f, nil
}

func (fb *FeatureBuilder) BuildFeatureVector(customerID int, snapshotDate time.Time) (map[string]float64, map[string]string, error) {
	featNum := make(map[string]float64)
	featCat := make(map[string]string)

	cust, err := fb.getCustomer(customerID)
	if err != nil {
		return nil, nil, err
	}

	featNum["age"] = float64(cust.Age)
	featNum["customer_lifetime_days"] = snapshotDate.Sub(cust.RegistrationDate).Hours() / 24
	featCat["city"] = cust.City
	featCat["gender"] = cust.Gender
	featCat["preferred_payment"] = cust.PreferredPayment

	ordFeat, err := fb.orderFeatures(customerID, snapshotDate)
	if err != nil {
		return nil, nil, err
	}
	for k, v := range ordFeat {
		featNum[k] = v
	}

	visFeat, err := fb.visitFeatures(customerID, snapshotDate)
	if err != nil {
		return nil, nil, err
	}
	for k, v := range visFeat {
		featNum[k] = v
	}

	supFeat, err := fb.supportFeatures(customerID, snapshotDate)
	if err != nil {
		return nil, nil, err
	}
	for k, v := range supFeat {
		featNum[k] = v
	}

	// дозаполнение отсутствующих числовых признаков нулями
	expected := []string{
		"age", "customer_lifetime_days",
		"orders_total", "orders_amount_total", "orders_amount_mean", "orders_quantity_total",
		"delivered_orders_total", "delivered_amount_total", "delivered_amount_mean",
		"orders_cancelled_total",
		"delivered_orders_30d", "delivered_amount_30d",
		"delivered_orders_90d", "delivered_amount_90d",
		"days_since_last_order",
		"visits_total", "pages_viewed_total", "pages_viewed_mean",
		"cart_adds_total", "checkout_initiated_total",
		"session_duration_mean",
		"visits_30d", "cart_adds_30d", "checkout_initiated_30d",
		"visits_90d", "cart_adds_90d", "checkout_initiated_90d",
		"days_since_last_visit",
		"support_tickets_total", "support_rating_mean",
		"support_resolution_days_mean",
		"support_tickets_90d", "days_since_last_ticket",
	}
	for _, name := range expected {
		if _, ok := featNum[name]; !ok {
			featNum[name] = 0
		}
	}
	// округление
	for k, v := range featNum {
		featNum[k] = math.Round(v*1000) / 1000
	}
	return featNum, featCat, nil
}
