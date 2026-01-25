import random
from typing import List, Dict, Any

def generate_quiz(quiz_type: str, difficulty: str, num_questions: int) -> List[Dict[str, Any]]:
    """Generate chemistry quiz questions."""
    questions = []
    
    if quiz_type == "balancing":
        questions = generate_balancing_questions(difficulty, num_questions)
    elif quiz_type == "elements":
        questions = generate_element_questions(difficulty, num_questions)
    elif quiz_type == "compounds":
        questions = generate_compound_questions(difficulty, num_questions)
    elif quiz_type == "reactions":
        questions = generate_reaction_type_questions(difficulty, num_questions)
    else:
        # Mixed quiz
        questions = generate_mixed_questions(difficulty, num_questions)
    
    return questions

def generate_balancing_questions(difficulty: str, count: int) -> List[Dict[str, Any]]:
    """Generate equation balancing questions."""
    equations = {
        "easy": [
            {"question": "Balance: H₂ + O₂ → H₂O", "answer": "2H2 + O2 → 2H2O", "type": "text"},
            {"question": "Balance: N₂ + H₂ → NH₃", "answer": "N2 + 3H2 → 2NH3", "type": "text"},
            {"question": "Balance: Na + Cl₂ → NaCl", "answer": "2Na + Cl2 → 2NaCl", "type": "text"},
            {"question": "Balance: Mg + O₂ → MgO", "answer": "2Mg + O2 → 2MgO", "type": "text"},
            {"question": "Balance: Al + O₂ → Al₂O₃", "answer": "4Al + 3O2 → 2Al2O3", "type": "text"},
        ],
        "medium": [
            {"question": "Balance: CH₄ + O₂ → CO₂ + H₂O", "answer": "CH4 + 2O2 → CO2 + 2H2O", "type": "text"},
            {"question": "Balance: Fe + O₂ → Fe₂O₃", "answer": "4Fe + 3O2 → 2Fe2O3", "type": "text"},
            {"question": "Balance: C₃H₈ + O₂ → CO₂ + H₂O", "answer": "C3H8 + 5O2 → 3CO2 + 4H2O", "type": "text"},
            {"question": "Balance: KClO₃ → KCl + O₂", "answer": "2KClO3 → 2KCl + 3O2", "type": "text"},
            {"question": "Balance: H₂SO₄ + NaOH → Na₂SO₄ + H₂O", "answer": "H2SO4 + 2NaOH → Na2SO4 + 2H2O", "type": "text"},
        ],
        "hard": [
            {"question": "Balance: C₆H₁₂O₆ + O₂ → CO₂ + H₂O", "answer": "C6H12O6 + 6O2 → 6CO2 + 6H2O", "type": "text"},
            {"question": "Balance: NH₃ + O₂ → NO + H₂O", "answer": "4NH3 + 5O2 → 4NO + 6H2O", "type": "text"},
            {"question": "Balance: Fe₂O₃ + CO → Fe + CO₂", "answer": "Fe2O3 + 3CO → 2Fe + 3CO2", "type": "text"},
            {"question": "Balance: Al + HCl → AlCl₃ + H₂", "answer": "2Al + 6HCl → 2AlCl3 + 3H2", "type": "text"},
            {"question": "Balance: Cu + HNO₃ → Cu(NO₃)₂ + NO + H₂O", "answer": "3Cu + 8HNO3 → 3Cu(NO3)2 + 2NO + 4H2O", "type": "text"},
        ]
    }
    
    pool = equations.get(difficulty, equations["medium"])
    selected = random.sample(pool, min(count, len(pool)))
    
    # Add multiple choice variants for some questions
    for i, q in enumerate(selected[:3]):  # First 3 as multiple choice
        q["type"] = "multiple_choice"
        q["options"] = generate_wrong_answers(q["answer"], "balancing")
        q["options"].append(q["answer"])
        random.shuffle(q["options"])
        q["explanation"] = f"The balanced equation is {q['answer']}"
    
    return selected

def generate_element_questions(difficulty: str, count: int) -> List[Dict[str, Any]]:
    """Generate element-related questions."""
    elements = {
        "H": "Hydrogen", "He": "Helium", "Li": "Lithium", "Be": "Beryllium",
        "B": "Boron", "C": "Carbon", "N": "Nitrogen", "O": "Oxygen",
        "F": "Fluorine", "Ne": "Neon", "Na": "Sodium", "Mg": "Magnesium",
        "Al": "Aluminum", "Si": "Silicon", "P": "Phosphorus", "S": "Sulfur",
        "Cl": "Chlorine", "K": "Potassium", "Ca": "Calcium", "Fe": "Iron",
        "Cu": "Copper", "Zn": "Zinc", "Ag": "Silver", "Au": "Gold",
        "Hg": "Mercury", "Pb": "Lead", "U": "Uranium"
    }
    
    questions = []
    element_list = list(elements.items())
    random.shuffle(element_list)
    
    for i in range(min(count, len(element_list))):
        symbol, name = element_list[i]
        
        if difficulty == "easy":
            # Symbol to name
            question = f"What is the name of the element with symbol {symbol}?"
            answer = name
            options = generate_wrong_answers(name, "element_name")
            question_type = "multiple_choice"
        elif difficulty == "medium":
            # Name to symbol
            question = f"What is the symbol for {name}?"
            answer = symbol
            options = generate_wrong_answers(symbol, "element_symbol")
            question_type = "multiple_choice"
        else:
            # Mixed: either symbol to name or name to symbol
            if random.choice([True, False]):
                question = f"What is the name of the element with symbol {symbol}?"
                answer = name
                options = generate_wrong_answers(name, "element_name")
            else:
                question = f"What is the symbol for {name}?"
                answer = symbol
                options = generate_wrong_answers(symbol, "element_symbol")
            question_type = "multiple_choice"
        
        options.append(answer)
        random.shuffle(options)
        
        questions.append({
            "question": question,
            "answer": answer,
            "type": question_type,
            "options": options,
            "explanation": f"{symbol} is the symbol for {name}"
        })
    
    return questions

def generate_compound_questions(difficulty: str, count: int) -> List[Dict[str, Any]]:
    """Generate compound-related questions."""
    compounds = {
        "easy": [
            {"formula": "H₂O", "name": "Water"},
            {"formula": "CO₂", "name": "Carbon Dioxide"},
            {"formula": "NaCl", "name": "Sodium Chloride"},
            {"formula": "HCl", "name": "Hydrochloric Acid"},
            {"formula": "H₂SO₄", "name": "Sulfuric Acid"},
        ],
        "medium": [
            {"formula": "NaOH", "name": "Sodium Hydroxide"},
            {"formula": "CH₄", "name": "Methane"},
            {"formula": "NH₃", "name": "Ammonia"},
            {"formula": "CaCO₃", "name": "Calcium Carbonate"},
            {"formula": "C₆H₁₂O₆", "name": "Glucose"},
        ],
        "hard": [
            {"formula": "CH₃COOH", "name": "Acetic Acid"},
            {"formula": "NaHCO₃", "name": "Sodium Bicarbonate"},
            {"formula": "KMnO₄", "name": "Potassium Permanganate"},
            {"formula": "C₂H₅OH", "name": "Ethanol"},
            {"formula": "H₃PO₄", "name": "Phosphoric Acid"},
        ]
    }
    
    pool = compounds.get(difficulty, compounds["medium"])
    selected = random.sample(pool, min(count, len(pool)))
    questions = []
    
    for compound in selected:
        if random.choice([True, False]):
            # Formula to name
            question = f"What is the name of the compound {compound['formula']}?"
            answer = compound["name"]
            options = generate_wrong_answers(compound["name"], "compound_name")
        else:
            # Name to formula
            question = f"What is the formula for {compound['name']}?"
            answer = compound["formula"]
            options = generate_wrong_answers(compound["formula"], "compound_formula")
        
        options.append(answer)
        random.shuffle(options)
        
        questions.append({
            "question": question,
            "answer": answer,
            "type": "multiple_choice",
            "options": options,
            "explanation": f"{compound['formula']} is {compound['name']}"
        })
    
    return questions

def generate_reaction_type_questions(difficulty: str, count: int) -> List[Dict[str, Any]]:
    """Generate reaction type identification questions."""
    reactions = {
        "easy": [
            {"equation": "2H₂ + O₂ → 2H₂O", "type": "Combustion/Synthesis"},
            {"equation": "HCl + NaOH → NaCl + H₂O", "type": "Acid-Base"},
            {"equation": "Zn + 2HCl → ZnCl₂ + H₂", "type": "Single Replacement"},
            {"equation": "2H₂O → 2H₂ + O₂", "type": "Decomposition"},
            {"equation": "AgNO₃ + NaCl → AgCl + NaNO₃", "type": "Double Replacement"},
        ],
        "medium": [
            {"equation": "CH₄ + 2O₂ → CO₂ + 2H₂O", "type": "Combustion"},
            {"equation": "2Na + Cl₂ → 2NaCl", "type": "Synthesis"},
            {"equation": "CaCO₃ → CaO + CO₂", "type": "Decomposition"},
            {"equation": "Cu + 2AgNO₃ → Cu(NO₃)₂ + 2Ag", "type": "Single Replacement"},
            {"equation": "BaCl₂ + Na₂SO₄ → BaSO₄ + 2NaCl", "type": "Double Replacement"},
        ],
        "hard": [
            {"equation": "C₆H₁₂O₆ + 6O₂ → 6CO₂ + 6H₂O", "type": "Combustion"},
            {"equation": "4NH₃ + 5O₂ → 4NO + 6H₂O", "type": "Redox"},
            {"equation": "2KClO₃ → 2KCl + 3O₂", "type": "Decomposition"},
            {"equation": "Fe₂O₃ + 3CO → 2Fe + 3CO₂", "type": "Redox"},
            {"equation": "H₂SO₄ + 2NaOH → Na₂SO₄ + 2H₂O", "type": "Acid-Base"},
        ]
    }
    
    pool = reactions.get(difficulty, reactions["medium"])
    selected = random.sample(pool, min(count, len(pool)))
    questions = []
    
    reaction_types = ["Combustion", "Synthesis", "Decomposition", 
                     "Single Replacement", "Double Replacement", "Acid-Base", "Redox"]
    
    for reaction in selected:
        question = f"What type of reaction is: {reaction['equation']}?"
        answer = reaction["type"]
        
        # Generate wrong answers
        wrong_types = [t for t in reaction_types if t not in answer]
        options = random.sample(wrong_types, 3)
        options.append(answer)
        random.shuffle(options)
        
        questions.append({
            "question": question,
            "answer": answer,
            "type": "multiple_choice",
            "options": options,
            "explanation": f"This is a {answer} reaction"
        })
    
    return questions

def generate_mixed_questions(difficulty: str, count: int) -> List[Dict[str, Any]]:
    """Generate mixed chemistry questions."""
    all_questions = []
    all_questions.extend(generate_balancing_questions(difficulty, count//4))
    all_questions.extend(generate_element_questions(difficulty, count//4))
    all_questions.extend(generate_compound_questions(difficulty, count//4))
    all_questions.extend(generate_reaction_type_questions(difficulty, count//4))
    
    # Fill remaining with random types
    remaining = count - len(all_questions)
    if remaining > 0:
        generators = [
            generate_balancing_questions,
            generate_element_questions,
            generate_compound_questions,
            generate_reaction_type_questions
        ]
        for _ in range(remaining):
            gen = random.choice(generators)
            q = gen(difficulty, 1)
            if q:
                all_questions.extend(q)
    
    random.shuffle(all_questions)
    return all_questions[:count]

def generate_wrong_answers(correct_answer: str, question_type: str) -> List[str]:
    """Generate plausible wrong answers for multiple choice questions."""
    wrong_answers = []
    
    if question_type == "balancing":
        # Generate wrong coefficients
        parts = correct_answer.split()
        for _ in range(3):
            wrong = []
            for part in parts:
                if part in ["→", "+"]:
                    wrong.append(part)
                else:
                    # Change coefficient
                    coeff = ''.join(filter(str.isdigit, part))
                    if coeff:
                        new_coeff = str(int(coeff) + random.choice([-1, 1, 2]))
                        if new_coeff == "0":
                            new_coeff = "1"
                        wrong.append(part.replace(coeff, new_coeff))
                    else:
                        wrong.append(part)
            wrong_answers.append(' '.join(wrong))
    
    elif question_type in ["element_name", "compound_name"]:
        # Wrong names
        common_names = ["Water", "Carbon", "Oxygen", "Hydrogen", "Nitrogen",
                       "Sodium", "Chlorine", "Calcium", "Iron", "Copper",
                       "Glucose", "Methane", "Ammonia", "Acetic Acid"]
        wrong_names = [n for n in common_names if n != correct_answer]
        wrong_answers = random.sample(wrong_names, 3)
    
    elif question_type in ["element_symbol", "compound_formula"]:
        # Wrong symbols/formulas
        common_symbols = ["H", "O", "C", "N", "Na", "Cl", "Ca", "Fe", "Cu", "Ag"]
        common_formulas = ["H2O", "CO2", "NaCl", "HCl", "H2SO4", "NaOH", "CH4", "NH3"]
        
        pool = common_symbols if question_type == "element_symbol" else common_formulas
        wrong = [s for s in pool if s != correct_answer]
        wrong_answers = random.sample(wrong, 3)
    
    return wrong_answers

# Test the generator
if __name__ == "__main__":
    print("Testing Quiz Generator...")
    quiz = generate_quiz("balancing", "medium", 5)
    for i, q in enumerate(quiz, 1):
        print(f"\nQ{i}: {q['question']}")
        if q.get('options'):
            print(f"Options: {q['options']}")
        print(f"Answer: {q['answer']}")
    print("\n✓ Quiz generator is working!")