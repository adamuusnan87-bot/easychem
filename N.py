from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
import json
import sqlite3
from datetime import datetime
import traceback
import re
import io
from utils.pdf_generator import generate_pdf_report

# ... (previous imports and setup remain the same) ...

@app.route("/solutions", methods=["GET", "POST"])
def solution_calculator():
    """Solution concentration calculator."""
    result = {}
    
    if request.method == "POST":
        try:
            calculation_type = request.form.get("calculation_type", "molarity")
            
            if calculation_type == "molarity":
                # Molarity calculation: M = moles / liters
                moles = float(request.form.get("moles", 0))
                liters = float(request.form.get("liters", 0))
                
                if liters > 0:
                    molarity = moles / liters
                    result["molarity"] = round(molarity, 4)
                    result["calculation"] = "Molarity"
                    result["formula"] = f"M = n / V = {moles} mol / {liters} L"
                    
                    # Also calculate if mass is provided
                    if request.form.get("mass"):
                        mass = float(request.form.get("mass"))
                        formula = request.form.get("formula", "NaCl")
                        from chem_engine import calculate_molar_mass
                        mm = calculate_molar_mass(formula)
                        if mm > 0:
                            calculated_moles = mass / mm
                            result["calculated_moles"] = round(calculated_moles, 4)
                            result["molar_mass"] = mm
            
            elif calculation_type == "dilution":
                # Dilution: C1V1 = C2V2
                c1 = float(request.form.get("c1", 0))
                v1 = float(request.form.get("v1", 0))
                c2 = float(request.form.get("c2", 0))
                v2 = float(request.form.get("v2", 0))
                
                # Calculate missing value
                if not c1 and v1 and c2 and v2:
                    c1 = (c2 * v2) / v1
                    result["c1"] = round(c1, 4)
                    result["calculation"] = "Initial Concentration"
                    result["formula"] = f"C₁ = (C₂ × V₂) / V₁ = ({c2} M × {v2} mL) / {v1} mL"
                elif c1 and not v1 and c2 and v2:
                    v1 = (c2 * v2) / c1
                    result["v1"] = round(v1, 4)
                    result["calculation"] = "Initial Volume"
                    result["formula"] = f"V₁ = (C₂ × V₂) / C₁ = ({c2} M × {v2} mL) / {c1} M"
                elif c1 and v1 and not c2 and v2:
                    c2 = (c1 * v1) / v2
                    result["c2"] = round(c2, 4)
                    result["calculation"] = "Final Concentration"
                    result["formula"] = f"C₂ = (C₁ × V₁) / V₂ = ({c1} M × {v1} mL) / {v2} mL"
                elif c1 and v1 and c2 and not v2:
                    v2 = (c1 * v1) / c2
                    result["v2"] = round(v2, 4)
                    result["calculation"] = "Final Volume"
                    result["formula"] = f"V₂ = (C₁ × V₁) / C₂ = ({c1} M × {v1} mL) / {c2} M"
            
            elif calculation_type == "preparation":
                # Solution preparation
                target_conc = float(request.form.get("target_conc", 0))
                target_vol = float(request.form.get("target_vol", 0))
                solute_mass = float(request.form.get("solute_mass", 0))
                formula = request.form.get("formula", "NaCl")
                
                from chem_engine import calculate_molar_mass
                mm = calculate_molar_mass(formula)
                
                if mm > 0:
                    if solute_mass > 0:
                        # Calculate concentration from mass
                        moles = solute_mass / mm
                        if target_vol > 0:
                            conc = moles / (target_vol / 1000)  # Convert mL to L
                            result["concentration"] = round(conc, 4)
                            result["calculation"] = "Concentration from Mass"
                            result["formula"] = f"C = (mass / MM) / V = ({solute_mass} g / {mm} g/mol) / {target_vol/1000} L"
                    elif target_conc > 0 and target_vol > 0:
                        # Calculate mass needed
                        moles_needed = target_conc * (target_vol / 1000)
                        mass_needed = moles_needed * mm
                        result["mass_needed"] = round(mass_needed, 4)
                        result["calculation"] = "Mass Required"
                        result["formula"] = f"mass = C × V × MM = {target_conc} M × {target_vol/1000} L × {mm} g/mol"
            
            elif calculation_type == "conversion":
                # Unit conversions
                value = float(request.form.get("value", 0))
                from_unit = request.form.get("from_unit", "M")
                to_unit = request.form.get("to_unit", "mM")
                
                conversions = {
                    "M_to_mM": value * 1000,
                    "mM_to_M": value / 1000,
                    "M_to_uM": value * 1000000,
                    "uM_to_M": value / 1000000,
                    "gL_to_M": lambda v, f: v / calculate_molar_mass(f) if f else 0,
                    "M_to_gL": lambda v, f: v * calculate_molar_mass(f) if f else 0,
                }
                
                key = f"{from_unit}_to_{to_unit}"
                if key in conversions:
                    if callable(conversions[key]):
                        formula = request.form.get("formula", "NaCl")
                        result["converted"] = round(conversions[key](value, formula), 4)
                    else:
                        result["converted"] = round(conversions[key], 4)
                    result["calculation"] = "Unit Conversion"
                    result["formula"] = f"{value} {from_unit} → {result['converted']} {to_unit}"
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = f"Calculation Error: {str(e)}"
            traceback.print_exc()
    
    return render_template("solutions.html", result=result)

@app.route("/quiz", methods=["GET", "POST"])
def chemistry_quiz():
    """Interactive chemistry quiz generator."""
    quiz_results = {}
    
    if request.method == "POST":
        try:
            quiz_type = request.form.get("quiz_type", "balancing")
            difficulty = request.form.get("difficulty", "easy")
            num_questions = int(request.form.get("num_questions", 5))
            
            # Generate quiz questions
            from utils.quiz_generator import generate_quiz
            questions = generate_quiz(quiz_type, difficulty, num_questions)
            
            # Check answers if submitted
            if "submit_answers" in request.form:
                score = 0
                user_answers = {}
                
                for i in range(num_questions):
                    user_answer = request.form.get(f"answer_{i}", "")
                    correct_answer = questions[i]["answer"]
                    
                    # Check if answer is correct
                    is_correct = str(user_answer).strip().lower() == str(correct_answer).strip().lower()
                    if is_correct:
                        score += 1
                    
                    user_answers[i] = {
                        "user_answer": user_answer,
                        "correct_answer": correct_answer,
                        "is_correct": is_correct
                    }
                
                quiz_results = {
                    "score": score,
                    "total": num_questions,
                    "percentage": round((score / num_questions) * 100, 1),
                    "user_answers": user_answers,
                    "questions": questions,
                    "graded": True
                }
            else:
                # Just show the quiz
                quiz_results = {
                    "questions": questions,
                    "graded": False,
                    "quiz_type": quiz_type,
                    "difficulty": difficulty
                }
            
        except Exception as e:
            quiz_results["error"] = f"Quiz Error: {str(e)}"
    
    return render_template("quiz.html", quiz_results=quiz_results)

@app.route("/export/pdf", methods=["POST"])
def export_pdf():
    """Export calculation results as PDF."""
    try:
        data = request.json
        pdf_data = generate_pdf_report(data)
        
        return send_file(
            io.BytesIO(pdf_data),
            mimetype='application/pdf',
            as_attachment=True,
            download_name='chemistry_report.pdf'
        )
    except Exception as e:
        return jsonify({"error": str(e), "success": False})

@app.route("/export/json", methods=["POST"])
def export_json():
    """Export calculation results as JSON."""
    try:
        data = request.json
        return jsonify({
            "success": True,
            "data": data,
            "exported_at": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e), "success": False})

# ... (rest of your existing routes remain the same) ...