import random
from typing import List, Dict, Any

def generate_quiz(quiz_type: str, difficulty: str, num_questions: int) -> List[Dict[str, Any]]:
    """Generate comprehensive chemistry quiz questions."""
    questions = []
    
    if quiz_type == "balancing":
        questions = generate_balancing_questions(difficulty, num_questions)
    elif quiz_type == "elements":
        questions = generate_element_questions(difficulty, num_questions)
    elif quiz_type == "compounds":
        questions = generate_compound_questions(difficulty, num_questions)
    elif quiz_type == "reactions":
        questions = generate_reaction_type_questions(difficulty, num_questions)
    elif quiz_type == "molar_mass":
        questions = generate_molar_mass_questions(difficulty, num_questions)
    elif quiz_type == "stoichiometry":
        questions = generate_stoichiometry_questions(difficulty, num_questions)
    else:
        # Mixed quiz
        questions = generate_mixed_questions(difficulty, num_questions)
    
    return questions

def generate_balancing_questions(difficulty: str, count: int) -> List[Dict[str, Any]]:
    """Generate equation balancing questions with step-by-step explanations."""
    equations = {
        "easy": [
            {"question": "Balance: H₂ + O₂ → H₂O", "answer": "2H2 + O2 → 2H2O", "type": "text",
             "explanation": "1. Count atoms: Left has 2H, 2O; Right has 2H, 1O\n2. Balance oxygen: H2 + O2 → 2H2O\n3. Now hydrogen is unbalanced: 2H2 + O2 → 2H2O\n4. Verify: 4H, 2O = 4H, 2O ✓"},
            {"question": "Balance: N₂ + H₂ → NH₃", "answer": "N2 + 3H2 → 2NH3", "type": "text",
             "explanation": "1. Nitrogen: 2 on left, 1 on right → need 2NH3\n2. Hydrogen: 2 on left, 6 on right → need 3H2\n3. Final: N2 + 3H2 → 2NH3"},
            {"question": "Balance: Na + Cl₂ → NaCl", "answer": "2Na + Cl2 → 2NaCl", "type": "text",
             "explanation": "Chlorine is diatomic (Cl2), so we need even number of NaCl. 2Na + Cl2 → 2NaCl"},
        ],
        "medium": [
            {"question": "Balance: CH₄ + O₂ → CO₂ + H₂O", "answer": "CH4 + 2O2 → CO2 + 2H2O", "type": "text",
             "explanation": "1. Carbon is balanced (1=1)\n2. Hydrogen: 4 on left, 2 on right → need 2H2O\n3. Oxygen: 2 on left, 4 on right → need 2O2\n4. Final: CH4 + 2O2 → CO2 + 2H2O"},
            {"question": "Balance: Fe + O₂ → Fe₂O₃", "answer": "4Fe + 3O2 → 2Fe2O3", "type": "text",
             "explanation": "1. Iron: 1 on left, 2 on right → need 2Fe2O3 (4Fe total)\n2. Oxygen: 2 on left, 6 on right → need 3O2\n3. Final: 4Fe + 3O2 → 2Fe2O3"},
            {"question": "Balance: H₂SO₄ + NaOH → Na₂SO₄ + H₂O", "answer": "H2SO4 + 2NaOH → Na2SO4 + 2H2O", "type": "text",
             "explanation": "Sulfuric acid is diprotic, so it needs 2 NaOH to neutralize. This produces 2 water molecules."},
        ],
        "hard": [
            {"question": "Balance: C₆H₁₂O₆ + O₂ → CO₂ + H₂O", "answer": "C6H12O6 + 6O2 → 6CO2 + 6H2O", "type": "text",
             "explanation": "1. Carbon: 6 on left → 6CO2\n2. Hydrogen: 12 on left → 6H2O (12H)\n3. Oxygen: 6 + 12 = 18 on left; 12 + 6 = 18 on right ✓"},
            {"question": "Balance: NH₃ + O₂ → NO + H₂O", "answer": "4NH3 + 5O2 → 4NO + 6H2O", "type": "text",
             "explanation": "This is tricky! Use systematic approach:\n1. Balance N: 4NH3 → 4NO\n2. Balance H: 12H → 6H2O\n3. Balance O: 10O needed → 5O2\n4. Final: 4NH3 + 5O2 → 4NO + 6H2O"},
            {"question": "Balance: Cu + HNO₃ → Cu(NO₃)₂ + NO + H₂O", "answer": "3Cu + 8HNO3 → 3Cu(NO3)2 + 2NO + 4H2O", "type": "text",
             "explanation": "Redox reaction! Copper is oxidized, nitric acid is reduced. Requires careful electron balance."},
        ]
    }
    
    pool = equations.get(difficulty, equations["medium"])
    selected = random.sample(pool, min(count, len(pool)))
    
    return selected

def generate_element_questions(difficulty: str, count: int) -> List[Dict[str, Any]]:
    """Generate element-related questions."""
    elements_data = {
        "H": {"name": "Hydrogen", "atomic_number": 1, "atomic_mass": 1.008, "group": 1, "period": 1, "category": "nonmetal"},
        "He": {"name": "Helium", "atomic_number": 2, "atomic_mass": 4.0026, "group": 18, "period": 1, "category": "noble"},
        "Li": {"name": "Lithium", "atomic_number": 3, "atomic_mass": 6.94, "group": 1, "period": 2, "category": "alkali"},
        "Be": {"name": "Beryllium", "atomic_number": 4, "atomic_mass": 9.0122, "group": 2, "period": 2, "category": "alkaline"},
        "B": {"name": "Boron", "atomic_number": 5, "atomic_mass": 10.81, "group": 13, "period": 2, "category": "semimetal"},
        "C": {"name": "Carbon", "atomic_number": 6, "atomic_mass": 12.011, "group": 14, "period": 2, "category": "nonmetal"},
        "N": {"name": "Nitrogen", "atomic_number": 7, "atomic_mass": 14.007, "group": 15, "period": 2, "category": "nonmetal"},
        "O": {"name": "Oxygen", "atomic_number": 8, "atomic_mass": 16.00, "group": 16, "period": 2, "category": "nonmetal"},
        "F": {"name": "Fluorine", "atomic_number": 9, "atomic_mass": 19.00, "group": 17, "period": 2, "category": "halogen"},
        "Ne": {"name": "Neon", "atomic_number": 10, "atomic_mass": 20.180, "group": 18, "period": 2, "category": "noble"},
        "Na": {"name": "Sodium", "atomic_number": 11, "atomic_mass": 22.990, "group": 1, "period": 3, "category": "alkali"},
        "Mg": {"name": "Magnesium", "atomic_number": 12, "atomic_mass": 24.305, "group": 2, "period": 3, "category": "alkaline"},
        "Al": {"name": "Aluminum", "atomic_number": 13, "atomic_mass": 26.982, "group": 13, "period": 3, "category": "basic"},
        "Si": {"name": "Silicon", "atomic_number": 14, "atomic_mass": 28.085, "group": 14, "period": 3, "category": "semimetal"},
        "P": {"name": "Phosphorus", "atomic_number": 15, "atomic_mass": 30.974, "group": 15, "period": 3, "category": "nonmetal"},
        "S": {"name": "Sulfur", "atomic_number": 16, "atomic_mass": 32.06, "group": 16, "period": 3, "category": "nonmetal"},
        "Cl": {"name": "Chlorine", "atomic_number": 17, "atomic_mass": 35.45, "group": 17, "period": 3, "category": "halogen"},
        "Ar": {"name": "Argon", "atomic_number": 18, "atomic_mass": 39.948, "group": 18, "period": 3, "category": "noble"},
        "K": {"name": "Potassium", "atomic_number": 19, "atomic_mass": 39.098, "group": 1, "period": 4, "category": "alkali"},
        "Ca": {"name": "Calcium", "atomic_number": 20, "atomic_mass": 40.078, "group": 2, "period": 4, "category": "alkaline"},
        "Fe": {"name": "Iron", "atomic_number": 26, "atomic_mass": 55.845, "group": 8, "period": 4, "category": "transition"},
        "Cu": {"name": "Copper", "atomic_number": 29, "atomic_mass": 63.546, "group": 11, "period": 4, "category": "transition"},
        "Zn": {"name": "Zinc", "atomic_number": 30, "atomic_mass": 65.38, "group": 12, "period": 4, "category": "transition"},
        "Ag": {"name": "Silver", "atomic_number": 47, "atomic_mass": 107.87, "group": 11, "period": 5, "category": "transition"},
        "I": {"name": "Iodine", "atomic_number": 53, "atomic_mass": 126.90, "group": 17, "period": 5, "category": "halogen"},
        "Ba": {"name": "Barium", "atomic_number": 56, "atomic_mass": 137.33, "group": 2, "period": 6, "category": "alkaline"},
        "Au": {"name": "Gold", "atomic_number": 79, "atomic_mass": 196.97, "group": 11, "period": 6, "category": "transition"},
        "Hg": {"name": "Mercury", "atomic_number": 80, "atomic_mass": 200.59, "group": 12, "period": 6, "category": "transition"},
        "Pb": {"name": "Lead", "atomic_number": 82, "atomic_mass": 207.2, "group": 14, "period": 6, "category": "basic"},
        "U": {"name": "Uranium", "atomic_number": 92, "atomic_mass": 238.03, "group": 3, "period": 7, "category": "actinide"}
    }
    
    questions = []
    element_list = list(elements_data.items())
    random.shuffle(element_list)
    
    for i in range(min(count, len(element_list))):
        symbol, data = element_list[i]
        
        question_types = [
            ("symbol_to_name", f"What is the name of the element with symbol {symbol}?", data["name"]),
            ("name_to_symbol", f"What is the symbol for {data['name']}?", symbol),
            ("atomic_number", f"What is the atomic number of {data['name']}?", str(data["atomic_number"])),
            ("atomic_mass", f"What is the atomic mass of {data['name']}? (Round to 2 decimals)", f"{data['atomic_mass']:.2f}"),
            ("group_period", f"What group and period is {data['name']} in?", f"Group {data['group']}, Period {data['period']}"),
            ("category", f"What category does {data['name']} belong to?", data["category"].replace('_', ' ').title())
        ]
        
        if difficulty == "easy":
            q_type = random.choice(question_types[:2])
        elif difficulty == "medium":
            q_type = random.choice(question_types[:4])
        else:
            q_type = random.choice(question_types)
        
        questions.append({
            "question": q_type[1],
            "answer": q_type[2],
            "type": "text",
            "explanation": f"{symbol} is {data['name']}, atomic number {data['atomic_number']}, atomic mass {data['atomic_mass']} u."
        })
    
    return questions

def generate_compound_questions(difficulty: str, count: int) -> List[Dict[str, Any]]:
    """Generate compound-related questions."""
    compounds = {
        "easy": [
            {"formula": "H₂O", "name": "Water", "type": "oxide"},
            {"formula": "CO₂", "name": "Carbon Dioxide", "type": "oxide"},
            {"formula": "NaCl", "name": "Sodium Chloride", "type": "salt"},
            {"formula": "HCl", "name": "Hydrochloric Acid", "type": "acid"},
            {"formula": "NaOH", "name": "Sodium Hydroxide", "type": "base"}
        ],
        "medium": [
            {"formula": "CH₄", "name": "Methane", "type": "hydrocarbon"},
            {"formula": "NH₃", "name": "Ammonia", "type": "base"},
            {"formula": "CaCO₃", "name": "Calcium Carbonate", "type": "salt"},
            {"formula": "H₂SO₄", "name": "Sulfuric Acid", "type": "acid"},
            {"formula": "C₂H₅OH", "name": "Ethanol", "type": "alcohol"}
        ],
        "hard": [
            {"formula": "CH₃COOH", "name": "Acetic Acid", "type": "carboxylic acid"},
            {"formula": "KMnO₄", "name": "Potassium Permanganate", "type": "oxidizing agent"},
            {"formula": "NaHCO₃", "name": "Sodium Bicarbonate", "type": "salt"},
            {"formula": "C₆H₁₂O₆", "name": "Glucose", "type": "carbohydrate"},
            {"formula": "H₃PO₄", "name": "Phosphoric Acid", "type": "acid"}
        ]
    }
    
    pool = compounds.get(difficulty, compounds["medium"])
    selected = random.sample(pool, min(count, len(pool)))
    questions = []
    
    for compound in selected:
        question_types = [
            ("formula_to_name", f"What is the name of the compound {compound['formula']}?", compound["name"]),
            ("name_to_formula", f"What is the formula for {compound['name']}?", compound["formula"]),
            ("identify_type", f"What type of compound is {compound['name']}?", compound["type"])
        ]
        
        q_type = random.choice(question_types)
        questions.append({
            "question": q_type[1],
            "answer": q_type[2],
            "type": "text",
            "explanation": f"{compound['formula']} is {compound['name']}, which is a {compound['type']}."
        })
    
    return questions

def generate_reaction_type_questions(difficulty: str, count: int) -> List[Dict[str, Any]]:
    """Generate reaction type identification questions."""
    reactions = {
        "easy": [
            {"equation": "2H₂ + O₂ → 2H₂O", "type": "Combustion", "category": "combustion"},
            {"equation": "HCl + NaOH → NaCl + H₂O", "type": "Acid-Base Neutralization", "category": "acid-base"},
            {"equation": "Zn + 2HCl → ZnCl₂ + H₂", "type": "Single Replacement", "category": "single_replacement"},
            {"equation": "2H₂O → 2H₂ + O₂", "type": "Decomposition", "category": "decomposition"},
            {"equation": "AgNO₃ + NaCl → AgCl + NaNO₃", "type": "Double Replacement", "category": "double_replacement"}
        ],
        "medium": [
            {"equation": "CH₄ + 2O₂ → CO₂ + 2H₂O", "type": "Combustion", "category": "combustion"},
            {"equation": "2Na + Cl₂ → 2NaCl", "type": "Synthesis", "category": "synthesis"},
            {"equation": "CaCO₃ → CaO + CO₂", "type": "Decomposition", "category": "decomposition"},
            {"equation": "Cu + 2AgNO₃ → Cu(NO₃)₂ + 2Ag", "type": "Single Replacement", "category": "single_replacement"},
            {"equation": "BaCl₂ + Na₂SO₄ → BaSO₄ + 2NaCl", "type": "Double Replacement (Precipitation)", "category": "precipitation"}
        ],
        "hard": [
            {"equation": "C₆H₁₂O₆ + 6O₂ → 6CO₂ + 6H₂O", "type": "Combustion", "category": "combustion"},
            {"equation": "4NH₃ + 5O₂ → 4NO + 6H₂O", "type": "Redox", "category": "redox"},
            {"equation": "2KClO₃ → 2KCl + 3O₂", "type": "Decomposition", "category": "decomposition"},
            {"equation": "Fe₂O₃ + 3CO → 2Fe + 3CO₂", "type": "Redox", "category": "redox"},
            {"equation": "H₂SO₄ + 2NaOH → Na₂SO₄ + 2H₂O", "type": "Acid-Base Neutralization", "category": "acid-base"}
        ]
    }
    
    pool = reactions.get(difficulty, reactions["medium"])
    selected = random.sample(pool, min(count, len(pool)))
    questions = []
    
    for reaction in selected:
        questions.append({
            "question": f"What type of reaction is: {reaction['equation']}?",
            "answer": reaction["type"],
            "type": "text",
            "explanation": f"This is a {reaction['type']} reaction because {get_reaction_explanation(reaction['category'])}."
        })
    
    return questions

def get_reaction_explanation(category):
    explanations = {
        "combustion": "a substance reacts with oxygen to produce heat and light",
        "acid-base": "an acid and base react to form salt and water",
        "single_replacement": "one element replaces another in a compound",
        "double_replacement": "ions exchange between two compounds",
        "precipitation": "two aqueous solutions form an insoluble solid",
        "decomposition": "a single compound breaks down into simpler substances",
        "synthesis": "two or more substances combine to form one product",
        "redox": "electrons are transferred between species"
    }
    return explanations.get(category, "of the specified type")

def generate_molar_mass_questions(difficulty: str, count: int) -> List[Dict[str, Any]]:
    """Generate molar mass calculation questions."""
    compounds = [
        {"formula": "H2O", "mass": 18.015},
        {"formula": "CO2", "mass": 44.01},
        {"formula": "NaCl", "mass": 58.44},
        {"formula": "CH4", "mass": 16.043},
        {"formula": "C6H12O6", "mass": 180.156},
        {"formula": "H2SO4", "mass": 98.079},
        {"formula": "CaCO3", "mass": 100.087},
        {"formula": "NH3", "mass": 17.031}
    ]
    
    selected = random.sample(compounds, min(count, len(compounds)))
    questions = []
    
    for compound in selected:
        if difficulty == "easy":
            question = f"What is the molar mass of {compound['formula']}?"
            answer = f"{compound['mass']:.2f}"
        elif difficulty == "medium":
            question = f"Calculate the molar mass of {compound['formula']} (show work)."
            answer = f"{compound['mass']:.2f} g/mol"
        else:
            question = f"How many grams are in 2.5 moles of {compound['formula']}?"
            answer = f"{compound['mass'] * 2.5:.2f}"
        
        questions.append({
            "question": question,
            "answer": answer,
            "type": "text",
            "explanation": f"Molar mass of {compound['formula']} = {compound['mass']:.2f} g/mol"
        })
    
    return questions

def generate_stoichiometry_questions(difficulty: str, count: int) -> List[Dict[str, Any]]:
    """Generate stoichiometry questions."""
    problems = [
        {
            "equation": "2H2 + O2 → 2H2O",
            "given": "5.0 g of H2",
            "find": "grams of H2O produced",
            "answer": "44.6",
            "explanation": "5.0 g H2 × (1 mol H2/2.016 g) × (2 mol H2O/2 mol H2) × (18.015 g H2O/1 mol) = 44.6 g H2O"
        },
        {
            "equation": "N2 + 3H2 → 2NH3",
            "given": "10.0 g of N2",
            "find": "grams of NH3 produced",
            "answer": "12.2",
            "explanation": "10.0 g N2 × (1 mol N2/28.02 g) × (2 mol NH3/1 mol N2) × (17.031 g NH3/1 mol) = 12.2 g NH3"
        },
        {
            "equation": "CaCO3 → CaO + CO2",
            "given": "25.0 g of CaCO3",
            "find": "grams of CO2 produced",
            "answer": "11.0",
            "explanation": "25.0 g CaCO3 × (1 mol CaCO3/100.09 g) × (1 mol CO2/1 mol CaCO3) × (44.01 g CO2/1 mol) = 11.0 g CO2"
        }
    ]
    
    selected = random.sample(problems, min(count, len(problems)))
    questions = []
    
    for problem in selected:
        if difficulty == "easy":
            question = f"For the reaction {problem['equation']}, if you start with {problem['given']}, how many {problem['find']}?"
        else:
            question = f"Stoichiometry Problem: {problem['equation']}\nStarting with {problem['given']}, calculate {problem['find']}. Show all steps."
        
        questions.append({
            "question": question,
            "answer": problem["answer"],
            "type": "text",
            "explanation": problem["explanation"]
        })
    
    return questions

def generate_mixed_questions(difficulty: str, count: int) -> List[Dict[str, Any]]:
    """Generate mixed chemistry questions."""
    all_questions = []
    generators = [
        generate_balancing_questions,
        generate_element_questions,
        generate_compound_questions,
        generate_reaction_type_questions,
        generate_molar_mass_questions,
        generate_stoichiometry_questions
    ]
    
    # Distribute questions evenly
    questions_per_type = count // len(generators)
    remainder = count % len(generators)
    
    for i, generator in enumerate(generators):
        num_q = questions_per_type + (1 if i < remainder else 0)
        if num_q > 0:
            all_questions.extend(generator(difficulty, num_q))
    
    random.shuffle(all_questions)
    return all_questions[:count]

# Test the generator
if __name__ == "__main__":
    print("Testing Quiz Generator...")
    quiz = generate_quiz("balancing", "medium", 3)
    for i, q in enumerate(quiz, 1):
        print(f"\nQ{i}: {q['question']}")
        print(f"Answer: {q['answer']}")
        print(f"Explanation: {q['explanation']}")
    print("\n✓ Quiz generator is working!")