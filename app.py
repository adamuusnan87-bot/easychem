from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
import json
import sqlite3
from datetime import datetime
import traceback
import re
import io
from utils.pdf_generator import generate_pdf_report

app = Flask(__name__)
app.secret_key = 'zamani-chem-secret-key-2024'

# Simple database initialization
def init_db():
    conn = sqlite3.connect('data/reactions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reactions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  equation TEXT,
                  balanced TEXT,
                  timestamp DATETIME)''')
    conn.commit()
    conn.close()

init_db()

def log_reaction(equation, balanced_eq):
    try:
        conn = sqlite3.connect('data/reactions.db')
        c = conn.cursor()
        c.execute("INSERT INTO reactions (equation, balanced, timestamp) VALUES (?, ?, ?)",
                 (equation, balanced_eq, datetime.now()))
        conn.commit()
        conn.close()
    except:
        pass

# Helper function to detect reaction type
def detect_reaction_type(equation):
    """Detect reaction type from equation string."""
    try:
        from chem_engine import get_reaction_type
        return get_reaction_type(equation)
    except:
        # Fallback simple detection
        equation_lower = equation.lower()
        
        if 'o2' in equation_lower and ('co2' in equation_lower or 'h2o' in equation_lower):
            return "Combustion"
        elif 'hcl' in equation_lower and 'naoh' in equation_lower:
            return "Acid-Base"
        elif 'zn' in equation_lower and 'hcl' in equation_lower:
            return "Single Replacement"
        elif 'agno3' in equation_lower and 'nacl' in equation_lower:
            return "Precipitation"
        elif 'caco3' in equation_lower and 'hcl' in equation_lower:
            return "Gas Evolution"
        elif '+' in equation_lower and '->' in equation_lower:
            left, right = equation_lower.split('->')
            left_parts = [p.strip() for p in left.split('+') if p.strip()]
            right_parts = [p.strip() for p in right.split('+') if p.strip()]
            
            if len(left_parts) > 1 and len(right_parts) == 1:
                return "Synthesis"
            elif len(left_parts) == 1 and len(right_parts) > 1:
                return "Decomposition"
        
        return "General"


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
@app.route("/", methods=["GET", "POST"])
def index():
    result = {}
    recent_reactions = []
    
    try:
        conn = sqlite3.connect('data/reactions.db')
        c = conn.cursor()
        c.execute("SELECT equation, balanced, timestamp FROM reactions ORDER BY timestamp DESC LIMIT 5")
        recent_reactions = c.fetchall()
        conn.close()
    except:
        pass
    
    if request.method == "POST":
        try:
            # Import chem engine functions
            from chem_engine import (
                balance_equation,
                ionic_equation,
                net_ionic,
                parse_reaction_string,
                ReactionComponent,
                calculate_molar_mass,
                get_reaction_type,
                calculate_enthalpy_change
            )
            
            equation = request.form["equation"]
            action = request.form.get("action", "balance")
            
            # Log reaction
            log_reaction(equation, "")
            
            if action == "predict":
                from chem_engine import predict_products
                predicted = predict_products(equation)
                result["predicted"] = predicted
                result["action"] = "prediction"
            else:
                # Parse and balance equation
                reaction_data = parse_reaction_string(equation)
                
                if reaction_data:
                    balanced, coeffs = balance_equation(equation)
                    
                    # Log balanced reaction
                    log_reaction(equation, balanced)
                    
                    result["balanced"] = balanced
                    result["ionic"] = ionic_equation(balanced)
                    result["net_ionic"] = net_ionic(result["ionic"])
                    result["reaction_type"] = get_reaction_type(balanced)
                    result["reactants"] = reaction_data["reactants"]
                    result["products"] = reaction_data["products"]
                    result["coefficients"] = coeffs
                    result["action"] = "balance"
                    
                    # Calculate molar masses
                    molar_masses = {}
                    enthalpies = {}
                    
                    for comp in reaction_data["reactants"] + reaction_data["products"]:
                        try:
                            mm = calculate_molar_mass(comp.formula)
                            molar_masses[comp.formula] = mm
                            
                            # Estimate enthalpy
                            enthalpy = calculate_enthalpy_change(comp.formula)
                            enthalpies[comp.formula] = enthalpy
                        except:
                            molar_masses[comp.formula] = "N/A"
                            enthalpies[comp.formula] = "N/A"
                    
                    result["molar_masses"] = molar_masses
                    result["enthalpies"] = enthalpies
                    
        except Exception as e:
            result["error"] = f"Error: {str(e)}"
            traceback.print_exc()

    return render_template("index.html", result=result, recent_reactions=recent_reactions)

@app.route("/yield", methods=["GET", "POST"])
def yield_calculator():
    result = {}
    if request.method == "POST":
        try:
            from chem_engine import (
                balance_equation,
                limiting_reagent,
                theoretical_yield,
                percent_yield,
                calculate_molar_mass
            )
            
            equation = request.form["equation"]
            unit = request.form.get("unit", "grams")
            actual_yield = request.form.get("actual_yield")
            
            # Collect reactant amounts
            reactants_data = {}
            for key in request.form:
                if key.startswith("reactant_"):
                    formula = key.replace("reactant_", "")
                    value = request.form[key]
                    if value.strip():
                        try:
                            reactants_data[formula] = float(value)
                        except:
                            pass
            
            if len(reactants_data) >= 2:
                reactants = list(reactants_data.keys())
                amounts = list(reactants_data.values())
                
                # Get balanced equation
                balanced, coeffs = balance_equation(equation)
                
                # Calculate limiting reagent
                limiting_amount, reagent_name, excess_reagents = limiting_reagent(
                    balanced, reactants, amounts
                )
                
                # Extract products from balanced equation
                products = []
                if '->' in balanced:
                    right_side = balanced.split('->')[1]
                    # Split by + and clean up
                    for comp in re.split(r'\s*\+\s*', right_side):
                        comp = comp.strip()
                        if comp:
                            # Remove coefficient to get formula
                            formula = re.sub(r'^\d+', '', comp)
                            if formula:
                                products.append(formula)
                
                yields = {}
                percent_yields = {}
                
                for product in products:
                    yield_amount = theoretical_yield(
                        balanced, 
                        product, 
                        limiting_amount, 
                        reagent_name
                    )
                    yields[product] = yield_amount
                    
                    # Calculate percent yield if actual yield provided
                    if actual_yield and product == request.form.get("target_product"):
                        try:
                            actual = float(actual_yield)
                            percent = percent_yield(actual, yield_amount)
                            percent_yields[product] = percent
                        except:
                            pass
                
                result["limiting"] = reagent_name
                result["limiting_amount"] = limiting_amount
                result["excess"] = excess_reagents
                result["theoretical_yields"] = yields
                result["percent_yields"] = percent_yields
                result["balanced_eq"] = balanced
                result["reactants_used"] = list(zip(reactants, amounts))
                result["unit"] = unit
                
        except Exception as e:
            result["error"] = f"Yield Calculation Error: {str(e)}"
            traceback.print_exc()
    
    return render_template("yield.html", result=result)

@app.route("/predict", methods=["GET", "POST"])
def predict_reaction():
    result = {}
    if request.method == "POST":
        try:
            from chem_engine import (
                predict_products,
                balance_equation,
                get_reaction_type,
                predict_precipitate,
                predict_gas_formation
            )
            
            reactants = request.form["reactants"]
            reaction_type = request.form.get("reaction_type", "auto")
            
            predicted_eq = predict_products(reactants, reaction_type)
            balanced, _ = balance_equation(predicted_eq)
            
            result["reactants"] = reactants
            result["predicted_equation"] = predicted_eq
            result["balanced_equation"] = balanced
            result["reaction_type"] = get_reaction_type(predicted_eq)
            result["precipitate"] = predict_precipitate(predicted_eq)
            result["gas_formed"] = predict_gas_formation(predicted_eq)
            
        except Exception as e:
            result["error"] = f"Prediction Error: {str(e)}"
    
    return render_template("predict.html", result=result)

@app.route("/database")
def reaction_database():
    reactions = []
    try:
        conn = sqlite3.connect('data/reactions.db')
        c = conn.cursor()
        c.execute("SELECT equation, balanced, timestamp FROM reactions ORDER BY timestamp DESC")
        reactions = c.fetchall()
        conn.close()
    except:
        pass
    
    # Pass the detect_reaction_type function to the template
    return render_template("database.html", reactions=reactions, detect_reaction_type=detect_reaction_type)

@app.route("/elements")
def periodic_table():
    try:
        with open('data/elements.json', 'r') as f:
            elements = json.load(f)
    except:
        from chem_engine import init_elements_data
        elements = init_elements_data()
    
    return render_template("elements.html", elements=elements)

@app.route("/api/balance", methods=["POST"])
def api_balance():
    try:
        data = request.json
        equation = data.get("equation", "")
        
        from chem_engine import balance_equation
        balanced, coeffs = balance_equation(equation)
        
        return jsonify({
            "balanced": balanced,
            "coefficients": coeffs,
            "success": True
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        })

@app.route("/api/calculate_mass", methods=["POST"])
def api_calculate_mass():
    try:
        data = request.json
        formula = data.get("formula", "")
        amount = float(data.get("amount", 1))
        
        from chem_engine import calculate_molar_mass, calculate_mass_from_moles
        molar_mass = calculate_molar_mass(formula)
        mass = calculate_mass_from_moles(amount, molar_mass)
        
        return jsonify({
            "molar_mass": molar_mass,
            "mass": mass,
            "formula": formula
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        })

@app.route("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "ZamaniChem Engine",
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    # Initialize elements data if needed
    try:
        with open('data/elements.json', 'r') as f:
            json.load(f)
    except:
        from chem_engine import init_elements_data
        init_elements_data()
    
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)