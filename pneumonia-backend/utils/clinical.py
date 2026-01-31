def calculate_clinical_metrics(probabilities: dict, prediction: str, confidence: float) -> dict:
    """
    Calculate clinical priority, severity, and alerts based on model predictions.
    
    Args:
        probabilities: Dict with 'Normal', 'Bacterial Pneumonia', 'Viral Pneumonia' probs
        prediction: Predicted class name
        confidence: Prediction confidence
    
    Returns:
        Dictionary with clinical metrics
    """
    # Extract probabilities
    bacterial_prob = probabilities.get("Bacterial Pneumonia", 0.0)
    viral_prob = probabilities.get("Viral Pneumonia", 0.0)
    
    # Calculate priority score (0-100)
    priority_score = int((bacterial_prob + viral_prob) * 100)
    
    # Determine urgency level
    if priority_score > 80:
        urgency_level = "CRITICAL"
        urgency_color = "#dc3545"  # Red
        urgency_icon = "🚨"
    elif priority_score > 60:
        urgency_level = "HIGH"
        urgency_color = "#fd7e14"  # Orange
        urgency_icon = "⚠️"
    elif priority_score > 40:
        urgency_level = "MEDIUM"
        urgency_color = "#ffc107"  # Yellow
        urgency_icon = "⚡"
    else:
        urgency_level = "LOW"
        urgency_color = "#28a745"  # Green
        urgency_icon = "✓"
    
    # Determine severity
    if prediction == "Normal":
        severity_level = "None"
        severity_description = "No pneumonia detected"
        lung_involvement = "None"
    elif confidence > 0.75:
        severity_level = "Severe"
        severity_description = "Significant consolidation present"
        lung_involvement = "Bilateral"
    elif confidence > 0.50:
        severity_level = "Moderate"
        severity_description = "Moderate infiltrates detected"
        lung_involvement = "Unilateral"
    else:
        severity_level = "Mild"
        severity_description = "Minor abnormalities present"
        lung_involvement = "Localized"
    
    # Alert system
    should_alert = priority_score > 85
    escalation_needed = priority_score > 95
    
    if should_alert:
        alert_message = "High-priority case requires prompt review"
    else:
        alert_message = None
    
    # Consultation recommendations
    consultation_recommended = False
    consultation_reason = None
    suggested_specialists = []
    
    if severity_level == "Severe":
        consultation_recommended = True
        consultation_reason = "Severe case requires multi-disciplinary review"
        suggested_specialists = ["Pulmonologist", "ICU Team"]
    elif confidence < 0.70 and prediction != "Normal":
        consultation_recommended = True
        consultation_reason = "Low confidence - second opinion recommended"
        suggested_specialists = ["Radiologist", "Pulmonologist"]
    
    return {
        "priority_score": priority_score,
        "urgency_level": urgency_level,
        "urgency_color": urgency_color,
        "urgency_icon": urgency_icon,
        "severity_level": severity_level,
        "severity_description": severity_description,
        "lung_involvement": lung_involvement,
        "should_alert": should_alert,
        "alert_message": alert_message,
        "escalation_needed": escalation_needed,
        "consultation_recommended": consultation_recommended,
        "consultation_reason": consultation_reason,
        "suggested_specialists": suggested_specialists
    }
