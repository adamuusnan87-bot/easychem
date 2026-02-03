from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
import json
import sqlite3
from datetime import datetime
import traceback
import re
import io
import os
from utils.pdf_generator import generate_pdf_report

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY') or 'zamani-chem-secret-key-2024'

# Security headers
@app.after_request
def add_security_headers(response):
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
        "img-src 'self'  https:; "
        "font-src 'self' https://cdnjs.cloudflare.com; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self';"
    )
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

# Database initialization
def init_db():
    os.makedirs('data', exist_ok=True)
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
    except Exception as e:
        print(f"DB log error: {e}")

def sanitize_chemical_input(text):
    """Sanitize user input for chemical equations."""
    if not text:
        return ""
    
    # Normalize common substitutions
    normalized = text.strip()
    normalized = normalized.replace('→', '->')  # Unicode arrow
    normalized = normalized.replace('=', '->')   # Equals sign
    normalized = normalized.replace(' ', '')     # Remove spaces
    
    # Handle subscripts that users might paste
    subscript_map = {
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
    }
    for sub, normal in subscript_map.items():
        normalized = normalized.replace(sub, normal)
    
    # Escape HTML and allow only safe characters
    import html
    safe_text = html.escape(normalized)
    allowed_pattern = r'[^A-Za-z0-9+\->(){}\[\]·•]'
    sanitized = re.sub(allowed_pattern, '', safe_text)
    
    return sanitized

def validate_chemical_equation(equation: str) -> bool:
    """Enhanced validation for chemical equations."""
    if not equation:
        return False
    
    # Must contain reaction arrow
    if '->' not in equation:
        return False
    
    # Must contain at least one element symbol (A-Z followed by optional a-z)
    if not re.search(r'[A-Z][a-z]?', equation):
        return False
    
    # Must have content on both sides of the arrow
    parts = equation.split('->')
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        return False
    
    # Must not contain invalid characters (except allowed chemical notation)
    allowed_pattern = r'^[A-Za-z0-9+\->(){}\[\]·•]+$'
    if not re.match(allowed_pattern, equation):
        return False
    
    return True

def parse_side(side: str):
    """Parse a side of the equation into coefficients and formulas."""
    compounds = []
    parts = side.split('+')
    for part in parts:
        part = part.strip()
        match = re.match(r'^(\d+)?(.*)$', part)
        if match:
            coeff_str = match.group(1)
            formula = match.group(2).strip()
            coeff = int(coeff_str) if coeff_str else 1
            if formula:
                compounds.append((coeff, formula))
    return compounds

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
    except Exception as e:
        print(f"DB error: {e}")
    
    if request.method == "POST":
        try:
            from chem_engine import (
                balance_equation,
                ionic_equation,
                net_ionic,
                parse_chemical_formula,
                calculate_molar_mass,
                get_reaction_type,
                calculate_enthalpy_change
            )
            
            # Get raw input for better error messages
            raw_equation = request.form.get("equation", "").strip()
            if not raw_equation:
                result["error"] = "Please enter a chemical equation."
                return render_template("index.html", result=result, recent_reactions=recent_reactions)
            
            # Check if user forgot the arrow
            if '->' not in raw_equation and '→' not in raw_equation and '=' not in raw_equation:
                if '+' in raw_equation:
                    result["error"] = "Chemical equations must have '->' between reactants and products. Example: H2 + O2 -> H2O"
                    return render_template("index.html", result=result, recent_reactions=recent_reactions)
                else:
                    result["error"] = "For single compounds, use the Solution Calculator. For reactions, use '->' like: H2 + O2 -> H2O"
                    return render_template("index.html", result=result, recent_reactions=recent_reactions)
            
            # Sanitize input
            equation = sanitize_chemical_input(raw_equation)
            
            if not validate_chemical_equation(equation):
                result["error"] = "Invalid chemical equation format. Please use '->' to separate reactants and products. Example: H2 + O2 -> H2O"
                return render_template("index.html", result=result, recent_reactions=recent_reactions)
            
            action = request.form.get("action", "balance")
            log_reaction(equation, "")
            
            if action == "predict":
                from chem_engine import predict_products
                predicted = predict_products(equation)
                result["predicted"] = predicted
                result["action"] = "prediction"
            else:
                balanced, coeffs = balance_equation(equation)
                
                if not balanced or '->' not in balanced:
                    result["error"] = "Could not balance this equation. Please check the format and try again."
                    return render_template("index.html", result=result, recent_reactions=recent_reactions)
                
                log_reaction(equation, balanced)
                
                result["balanced"] = balanced
                result["ionic"] = ionic_equation(balanced)
                result["net_ionic"] = net_ionic(result["ionic"])
                result["reaction_type"] = get_reaction_type(balanced)
                result["coefficients"] = coeffs
                result["action"] = "balance"
                
                # Calculate molar masses for all unique compounds
                molar_masses = {}
                enthalpies = {}
                
                # Extract all compounds from balanced equation
                def extract_compounds(equation_str):
                    compounds = set()
                    parts = equation_str.replace('->', '+').split('+')
                    for part in parts:
                        part = part.strip()
                        # Remove coefficient
                        match = re.match(r'^\d*(.+)$', part)
                        if match:
                            formula = match.group(1)
                            compounds.add(formula)
                    return list(compounds)
                
                all_compounds = extract_compounds(balanced)
                
                for compound in all_compounds:
                    try:
                        mm = calculate_molar_mass(compound)
                        molar_masses[compound] = mm
                        enthalpy = calculate_enthalpy_change(compound)
                        enthalpies[compound] = enthalpy
                    except:
                        molar_masses[compound] = "N/A"
                        enthalpies[compound] = "N/A"
                
                result["molar_masses"] = molar_masses
                result["enthalpies"] = enthalpies

                # Expanded: Calculate reaction enthalpy change (ΔH_rxn)
                try:
                    left, right = balanced.split('->')
                    reactants = parse_side(left)
                    products = parse_side(right)
                    
                    delta_h = 0.0
                    all_available = True
                    
                    # Sum for products
                    for coeff, formula in products:
                        hf = enthalpies.get(formula)
                        if not isinstance(hf, (int, float)):
                            all_available = False
                            break
                        delta_h += coeff * hf
                    
                    if all_available:
                        # Subtract for reactants
                        for coeff, formula in reactants:
                            hf = enthalpies.get(formula)
                            if not isinstance(hf, (int, float)):
                                all_available = False
                                break
                            delta_h -= coeff * hf
                    
                    if all_available:
                        result["delta_h"] = round(delta_h, 2)  # Round to 2 decimal places for display
                    else:
                        result["delta_h"] = "N/A"
                except Exception as e:
                    result["delta_h"] = "N/A"
                    print(f"Delta H calculation error: {e}")
                
        except Exception as e:
            result["error"] = f"Error processing equation: {str(e)}"
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
            
            raw_equation = request.form.get("equation", "").strip()
            if not raw_equation:
                result["error"] = "Please enter a chemical equation."
                return render_template("yield.html", result=result)
            
            equation = sanitize_chemical_input(raw_equation)
            if not validate_chemical_equation(equation):
                result["error"] = "Invalid chemical equation format."
                return render_template("yield.html", result=result)
            
            unit = request.form.get("unit", "grams")
            actual_yield = request.form.get("actual_yield")
            
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
                
                balanced, coeffs = balance_equation(equation)
                if not balanced or '->' not in balanced:
                    raise ValueError("Could not balance equation")
                
                limiting_amount, reagent_name, excess_reagents = limiting_reagent(
                    balanced, reactants, amounts
                )
                
                products = []
                if '->' in balanced:
                    right_side = balanced.split('->')[1]
                    for comp in re.split(r'\s*\+\s*', right_side):
                        comp = comp.strip()
                        if comp:
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

@app.route("/solutions", methods=["GET", "POST"])
def solution_calculator():
    result = {}
    if request.method == "POST":
        try:
            calculation_type = request.form.get("calculation_type", "molarity")
            
            if calculation_type == "molarity":
                moles = float(request.form.get("moles", 0))
                liters = float(request.form.get("liters", 0))
                
                if liters > 0:
                    molarity = moles / liters
                    result["molarity"] = round(molarity, 4)
                    result["calculation"] = "Molarity"
                    result["formula"] = f"M = n / V = {moles} mol / {liters} L"
                    
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
                c1 = float(request.form.get("c1", 0))
                v1 = float(request.form.get("v1", 0))
                c2 = float(request.form.get("c2", 0))
                v2 = float(request.form.get("v2", 0))
                
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
                target_conc = float(request.form.get("target_conc", 0))
                target_vol = float(request.form.get("target_vol", 0))
                solute_mass = float(request.form.get("solute_mass", 0))
                formula = request.form.get("formula", "NaCl")
                
                from chem_engine import calculate_molar_mass
                mm = calculate_molar_mass(formula)
                
                if mm > 0:
                    if solute_mass > 0:
                        moles = solute_mass / mm
                        if target_vol > 0:
                            conc = moles / (target_vol / 1000)
                            result["concentration"] = round(conc, 4)
                            result["calculation"] = "Concentration from Mass"
                            result["formula"] = f"C = (mass / MM) / V = ({solute_mass} g / {mm} g/mol) / {target_vol/1000} L"
                    elif target_conc > 0 and target_vol > 0:
                        moles_needed = target_conc * (target_vol / 1000)
                        mass_needed = moles_needed * mm
                        result["mass_needed"] = round(mass_needed, 4)
                        result["calculation"] = "Mass Required"
                        result["formula"] = f"mass = C × V × MM = {target_conc} M × {target_vol/1000} L × {mm} g/mol"
            
            elif calculation_type == "conversion":
                value = float(request.form.get("value", 0))
                from_unit = request.form.get("from_unit", "M")
                to_unit = request.form.get("to_unit", "mM")
                
                conversions = {
                    "M_to_mM": value * 1000,
                    "mM_to_M": value / 1000,
                    "M_to_uM": value * 1000000,
                    "uM_to_M": value / 1000000,
                }
                
                key = f"{from_unit}_to_{to_unit}"
                if key in conversions:
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
    quiz_results = {}
    if request.method == "POST":
        try:
            quiz_type = request.form.get("quiz_type", "balancing")
            difficulty = request.form.get("difficulty", "easy")
            num_questions = int(request.form.get("num_questions", 5))
            
            from utils.quiz_generator import generate_quiz
            questions = generate_quiz(quiz_type, difficulty, num_questions)
            
            if "submit_answers" in request.form:
                score = 0
                user_answers = {}
                
                for i in range(num_questions):
                    user_answer = request.form.get(f"answer_{i}", "")
                    correct_answer = questions[i]["answer"]
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
    try:
        data = request.json
        return jsonify({
            "success": True,
            "data": data,
            "exported_at": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e), "success": False})

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
            
            reactants = sanitize_chemical_input(request.form["reactants"])
            if not reactants:
                result["error"] = "Please enter reactants."
                return render_template("predict.html", result=result)
            
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
    except Exception as e:
        print(f"DB error: {e}")
    
    from chem_engine import detect_reaction_type
    return render_template("database.html", reactions=reactions, detect_reaction_type=detect_reaction_type)

@app.route("/elements")
def periodic_table_page():
    """Render the periodic table HTML page"""
    return render_template("elements.html")

@app.route("/api/elements")
def get_elements_data():
    """API endpoint that returns elements as JSON"""
    try:
        with open('data/elements.json', 'r') as f:
            elements = json.load(f)
    except:
        from chem_engine import init_elements_data
        elements = init_elements_data()
    
    return jsonify(elements)

@app.route("/molecule")
def molecule_viewer():
    formula = request.args.get('formula', 'H2O')
    return render_template("molecule.html", formula=formula)

@app.route("/ocr")
def ocr_scanner():
    return render_template("ocr.html")

@app.route("/mechanism")
def reaction_mechanism():
    return render_template("mechanism.html")

@app.route("/api/balance", methods=["POST"])
def api_balance():
    try:
        data = request.json
        equation = sanitize_chemical_input(data.get("equation", ""))
        if not validate_chemical_equation(equation):
            return jsonify({"error": "Invalid equation", "success": False})
        
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
    try:
        with open('data/elements.json', 'r') as f:
            json.load(f)
    except:
        from chem_engine import init_elements_data
        init_elements_data()
    
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)