import re
import json
import os
import html
from typing import List, Tuple, Dict, Optional, Union, Any, Set
from collections import defaultdict
from functools import lru_cache
from sympy import Matrix, lcm, gcd, nsimplify


class ChemicalEquationError(Exception):
    """Custom exception for chemical equation errors."""
    pass


class ReactionComponent:
    """Represents a chemical compound with its coefficient and state."""
    
    def __init__(self, formula: str, coefficient: int = 1, state: str = ''):
        self.formula = formula
        self.coefficient = coefficient
        self.state = state  # (s), (l), (g), (aq)
    
    def __repr__(self):
        coeff_str = str(self.coefficient) if self.coefficient != 1 else ''
        state_str = f"({self.state})" if self.state else ''
        return f"{coeff_str}{self.formula}{state_str}"
    
    def __eq__(self, other):
        if not isinstance(other, ReactionComponent):
            return False
        return (self.formula == other.formula and 
                self.coefficient == other.coefficient and 
                self.state == other.state)


def sanitize_chemical_input(text: str) -> str:
    """Sanitize user input for chemical equations with enhanced safety."""
    if not text:
        return ""
    
    # Normalize common substitutions
    normalized = text.strip()
    
    # Handle various arrow types
    arrow_replacements = [
        ('→', '->'), ('⟶', '->'), ('⇒', '->'), ('⇨', '->'),
        ('↦', '->'), ('↔', '<->'), ('⇌', '<->'), ('⇋', '<->')
    ]
    
    for old, new in arrow_replacements:
        normalized = normalized.replace(old, new)
    
    # Normalize equals sign
    if '=' in normalized and '->' not in normalized:
        normalized = normalized.replace('=', '->')
    
    # Remove spaces around arrows
    normalized = re.sub(r'\s*->\s*', '->', normalized)
    normalized = re.sub(r'\s*<->\s*', '<->', normalized)
    
    # Handle subscripts
    subscript_map = {
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
        '₊': '+', '₋': '-', '₌': '=', '₍': '(', '₎': ')'
    }
    
    for sub, normal in subscript_map.items():
        normalized = normalized.replace(sub, normal)
    
    # Handle superscripts (for charges)
    superscript_map = {
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
        '⁺': '+', '⁻': '-', '⁼': '=', '⁽': '(', '⁾': ')'
    }
    
    for sup, normal in superscript_map.items():
        normalized = normalized.replace(sup, normal)
    
    # Escape HTML
    safe_text = html.escape(normalized)
    
    # Allow only safe characters for chemical notation
    allowed_pattern = r'[^A-Za-z0-9\+\-\><\(\)\{\}\[\]·•\.\s]'
    sanitized = re.sub(allowed_pattern, '', safe_text)
    
    # Clean up multiple spaces
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
    return sanitized


def validate_chemical_equation(equation: str) -> Tuple[bool, str]:
    """Enhanced validation for chemical equations with detailed error messages."""
    if not equation:
        return False, "Empty equation"
    
    # Check for reaction arrow
    if '->' not in equation and '<->' not in equation:
        return False, "No reaction arrow found. Use '->' for irreversible or '<->' for reversible reactions"
    
    # Determine arrow type
    arrow = '->' if '->' in equation else '<->'
    
    # Split equation
    if arrow in equation:
        parts = equation.split(arrow)
        if len(parts) != 2:
            return False, f"Invalid equation format. Expected format: reactants {arrow} products"
        
        left, right = parts
        if not left.strip() or not right.strip():
            return False, "Both sides of the arrow must contain compounds"
    
    # Must contain at least one valid element symbol
    if not re.search(r'[A-Z][a-z]?\d*', equation):
        return False, "No valid chemical elements found"
    
    # Check for invalid characters (after sanitization, this should be rare)
    allowed_pattern = r'^[A-Za-z0-9\+\-\><\(\)\{\}\[\]·•\.\s]+$'
    if not re.match(allowed_pattern, equation):
        return False, "Invalid characters in equation"
    
    # Check for unbalanced parentheses/brackets
    paren_count = 0
    bracket_count = 0
    for char in equation:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        elif char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
    
    if paren_count != 0:
        return False, "Unbalanced parentheses"
    if bracket_count != 0:
        return False, "Unbalanced brackets"
    
    # Check for consecutive operators
    if re.search(r'[+\-]\s*[+\-]', equation):
        return False, "Consecutive operators found"
    
    return True, "Valid equation"


@lru_cache(maxsize=128)
def parse_chemical_formula(formula: str) -> Dict[str, int]:
    """
    Parse any chemical formula including:
    - Simple: H2O, CO2
    - Parentheses: Ca(OH)2, Al2(SO4)3  
    - Hydrates: CuSO4·5H2O
    - Organic: CH3COOH, C6H12O6
    - Complex: K4[Fe(CN)6]
    - Charges: SO4²⁻, NH4⁺
    """
    if not formula:
        return {}
    
    # Remove state symbols if present
    formula = re.sub(r'\([slgaq]+\)$', '', formula)
    
    # Handle hydrates (separate water molecules)
    if '·' in formula or '.' in formula:
        separator = '·' if '·' in formula else '.'
        parts = formula.split(separator)
        main_part = parts[0].strip()
        hydrate_parts = parts[1:]
        
        main_elements = parse_chemical_formula(main_part)
        combined = main_elements.copy()
        
        for hydrate_part in hydrate_parts:
            hydrate_part = hydrate_part.strip()
            # Extract multiplier from hydrate part (e.g., "5H2O" -> multiplier=5, formula="H2O")
            match = re.match(r'^(\d*)(.+)$', hydrate_part)
            if match:
                multiplier_str, hydrate_formula = match.groups()
                multiplier = int(multiplier_str) if multiplier_str else 1
                hydrate_elements = parse_chemical_formula(hydrate_formula)
                for element, count in hydrate_elements.items():
                    combined[element] = combined.get(element, 0) + (count * multiplier)
        
        return combined
    
    # Handle brackets [Fe(CN)6] -> treat as parentheses
    formula = formula.replace('[', '(').replace(']', ')')
    
    def parse_group(formula_part: str, multiplier: int = 1) -> Dict[str, int]:
        """Recursively parse formula groups with proper element validation."""
        element_counts = defaultdict(int)
        i = 0
        n = len(formula_part)
        
        # All 118 elements
        known_elements = {
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
            'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
            'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
            'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
        }
        
        while i < n:
            if formula_part[i] == '(':
                # Find matching closing parenthesis
                paren_count = 1
                j = i + 1
                while j < n and paren_count > 0:
                    if formula_part[j] == '(':
                        paren_count += 1
                    elif formula_part[j] == ')':
                        paren_count -= 1
                    j += 1
                
                if paren_count != 0:
                    # Unmatched parenthesis, skip to next character
                    i += 1
                    continue
                
                # Extract group content
                group_content = formula_part[i+1:j-1]
                
                # Get multiplier after parenthesis
                k = j
                num_str = ''
                while k < n and formula_part[k].isdigit():
                    num_str += formula_part[k]
                    k += 1
                
                group_multiplier = int(num_str) if num_str else 1
                total_multiplier = multiplier * group_multiplier
                
                # Recursively parse the group
                group_counts = parse_group(group_content, total_multiplier)
                for elem, count in group_counts.items():
                    element_counts[elem] += count
                
                i = k
            elif formula_part[i].isupper():
                # Parse element symbol (1-2 characters)
                element = formula_part[i]
                i += 1
                # Check for second character (lowercase)
                if i < n and formula_part[i].islower():
                    element += formula_part[i]
                    i += 1
                
                # Validate element
                if element not in known_elements:
                    # Skip invalid elements
                    i += 1
                    continue
                
                # Get subscript number
                num_str = ''
                while i < n and formula_part[i].isdigit():
                    num_str += formula_part[i]
                    i += 1
                
                count = int(num_str) if num_str else 1
                element_counts[element] += count * multiplier
            else:
                # Skip invalid characters
                i += 1
        
        return dict(element_counts)
    
    return parse_group(formula)


def parse_reaction_string(equation: str) -> Optional[Dict[str, List[ReactionComponent]]]:
    """Parse reaction string into structured data with state symbols."""
    try:
        equation = equation.strip()
        
        # Determine arrow type
        if '->' in equation:
            separator = '->'
            reactants_str, products_str = equation.split('->')
        elif '<->' in equation:
            separator = '<->'
            reactants_str, products_str = equation.split('<->')
        elif '=' in equation:
            separator = '='
            reactants_str, products_str = equation.split('=')
        else:
            return None
        
        def parse_compounds(comp_str: str) -> List[ReactionComponent]:
            """Extract compounds with coefficients and states from string."""
            compounds = []
            current = ''
            paren_count = 0
            bracket_count = 0
            
            # Split by + but respect parentheses and brackets
            for char in comp_str:
                if char == '(':
                    paren_count += 1
                    current += char
                elif char == ')':
                    paren_count -= 1
                    current += char
                elif char == '[':
                    bracket_count += 1
                    current += char
                elif char == ']':
                    bracket_count -= 1
                    current += char
                elif char == '+' and paren_count == 0 and bracket_count == 0:
                    if current.strip():
                        compounds.append(current.strip())
                    current = ''
                else:
                    current += char
            
            if current.strip():
                compounds.append(current.strip())
            
            # Parse each compound
            result = []
            for comp in compounds:
                if not comp:
                    continue
                
                # Extract coefficient, formula, and state
                # Pattern matches: coefficient, formula, state in parentheses
                match = re.match(r'^(\d*)([^()]+?)(?:\(([slgaq]+)\))?$', comp)
                if match:
                    coeff_str, formula, state = match.groups()
                    coeff = int(coeff_str) if coeff_str else 1
                    formula = formula.strip()
                    state = state if state else ''
                    
                    # Validate formula
                    if parse_chemical_formula(formula):
                        result.append(ReactionComponent(formula, coeff, state))
            
            return result
        
        reactants = parse_compounds(reactants_str)
        products = parse_compounds(products_str)
        
        if not reactants or not products:
            return None
        
        return {
            "reactants": reactants,
            "products": products,
            "separator": separator
        }
        
    except Exception as e:
        print(f"Error parsing reaction string: {e}")
        return None


@lru_cache(maxsize=64)
def balance_equation(equation: str) -> Tuple[str, Dict[str, int]]:
    """
    Balance ANY chemical equation using linear algebra.
    Returns balanced equation and coefficients dictionary.
    """
    try:
        # Clean and validate the equation
        equation = equation.strip().replace(' ', '')
        if not equation:
            return equation, {}
        
        # Determine arrow type
        if '->' in equation:
            arrow = '->'
            left, right = equation.split('->')
        elif '<->' in equation:
            arrow = '<->'
            left, right = equation.split('<->')
        elif '=' in equation:
            arrow = '->'
            left, right = equation.split('=')
        else:
            return equation, {}
        
        def parse_compounds(comp_str: str) -> List[str]:
            """Extract compound formulas from string."""
            compounds = []
            current = ''
            paren_count = 0
            bracket_count = 0
            
            for char in comp_str:
                if char == '(':
                    paren_count += 1
                    current += char
                elif char == ')':
                    paren_count -= 1
                    current += char
                elif char == '[':
                    bracket_count += 1
                    current += char
                elif char == ']':
                    bracket_count -= 1
                    current += char
                elif char == '+' and paren_count == 0 and bracket_count == 0:
                    if current.strip():
                        # Remove coefficient for parsing
                        comp = re.sub(r'^\d+', '', current.strip())
                        compounds.append(comp)
                    current = ''
                else:
                    current += char
            
            if current.strip():
                comp = re.sub(r'^\d+', '', current.strip())
                compounds.append(comp)
            
            return compounds
        
        reactants = parse_compounds(left)
        products = parse_compounds(right)
        
        if not reactants or not products:
            return equation, {}
        
        all_compounds = reactants + products
        
        # Get all unique elements from all compounds
        all_elements = set()
        compound_elements = {}
        
        for compound in all_compounds:
            elements = parse_chemical_formula(compound)
            if not elements:  # Skip invalid compounds
                continue
            compound_elements[compound] = elements
            all_elements.update(elements.keys())
        
        if not all_elements:
            return equation, {}
        
        all_elements = sorted(all_elements)
        
        # Build the composition matrix
        # Each row = element, each column = compound
        # Reactants are positive, products are negative
        matrix = []
        for element in all_elements:
            row = []
            # Reactants (positive coefficients)
            for compound in reactants:
                count = compound_elements.get(compound, {}).get(element, 0)
                row.append(count)
            # Products (negative coefficients)  
            for compound in products:
                count = compound_elements.get(compound, {}).get(element, 0)
                row.append(-count)
            matrix.append(row)
        
        # Solve using sympy's nullspace method
        M = Matrix(matrix)
        nullspace = M.nullspace()
        
        if not nullspace:
            return equation, {}
        
        # Get the solution vector
        solution = nullspace[0]
        
        # Convert to rational numbers
        rational_solution = []
        for val in solution:
            rational = nsimplify(val)
            rational_solution.append(rational)
        
        # Convert to integers by finding LCM of denominators
        denominators = []
        for val in rational_solution:
            if hasattr(val, 'as_numer_denom'):
                _, denom = val.as_numer_denom()
                denominators.append(abs(denom))
            else:
                denominators.append(1)
        
        lcm_denom = 1
        for denom in denominators:
            lcm_denom = lcm(lcm_denom, denom)
        
        # Multiply by LCM to get integers
        integer_solution = []
        for val in rational_solution:
            if hasattr(val, 'as_numer_denom'):
                numer, denom = val.as_numer_denom()
                integer_val = numer * (lcm_denom // denom)
            else:
                integer_val = val * lcm_denom
            integer_solution.append(int(integer_val))
        
        # Ensure all coefficients are positive
        if integer_solution[0] < 0:
            integer_solution = [-x for x in integer_solution]
        
        # Simplify by dividing by GCD
        def compute_gcd_of_list(numbers):
            result = abs(numbers[0])
            for num in numbers[1:]:
                result = gcd(result, abs(num))
            return result
        
        gcd_all = compute_gcd_of_list(integer_solution)
        if gcd_all > 1:
            integer_solution = [x // gcd_all for x in integer_solution]
        
        # Split coefficients for reactants and products
        reactant_coeffs = integer_solution[:len(reactants)]
        product_coeffs = integer_solution[len(reactants):]
        
        # Format the balanced equation
        def format_coefficient(coeff: int, formula: str) -> str:
            """Format coefficient with formula."""
            if coeff == 1:
                return formula
            return f"{coeff}{formula}"
        
        balanced_reactants = [
            format_coefficient(coeff, formula) 
            for coeff, formula in zip(reactant_coeffs, reactants)
        ]
        balanced_products = [
            format_coefficient(coeff, formula) 
            for coeff, formula in zip(product_coeffs, products)
        ]
        
        balanced_equation = " + ".join(balanced_reactants) + f" {arrow} " + " + ".join(balanced_products)
        
        # Create coefficient dictionary
        coeff_dict = {}
        for coeff, formula in zip(reactant_coeffs, reactants):
            coeff_dict[formula] = coeff
        for coeff, formula in zip(product_coeffs, products):
            coeff_dict[formula] = coeff
        
        return balanced_equation, coeff_dict
        
    except Exception as e:
        print(f"Balancing error: {e}")
        # Return formatted original equation as fallback
        formatted = equation.replace('+', ' + ').replace('->', ' -> ').replace('<->', ' <-> ').replace('=', ' = ')
        return formatted, {}


def validate_and_balance(equation: str) -> Tuple[str, Dict[str, int]]:
    """Validate and balance with detailed error messages."""
    is_valid, message = validate_chemical_equation(equation)
    
    if not is_valid:
        raise ChemicalEquationError(message)
    
    balanced, coeffs = balance_equation(equation)
    
    if not coeffs:
        raise ChemicalEquationError("Unable to balance equation. Please check if it's a valid chemical equation.")
    
    return balanced, coeffs


def ionic_equation(balanced_eq: str) -> str:
    """Generate complete ionic equation with state symbols."""
    # Common strong electrolytes that dissociate completely
    strong_acids = ['HCl', 'HBr', 'HI', 'HNO3', 'HClO3', 'HClO4', 'H2SO4']
    strong_bases = ['NaOH', 'KOH', 'LiOH', 'Ca(OH)2', 'Ba(OH)2', 'Sr(OH)2']
    soluble_salts = [
        'NaCl', 'KCl', 'NaNO3', 'KNO3', 'Na2SO4', 'K2SO4',
        'MgCl2', 'CaCl2', 'BaCl2', 'AgNO3', 'NH4Cl'
    ]
    
    # Parse the equation
    parsed = parse_reaction_string(balanced_eq)
    if not parsed:
        return balanced_eq
    
    # Ionic forms with state symbols
    ionic_forms = {
        # Strong acids
        'HCl': 'H⁺(aq) + Cl⁻(aq)',
        'HBr': 'H⁺(aq) + Br⁻(aq)',
        'HI': 'H⁺(aq) + I⁻(aq)',
        'HNO3': 'H⁺(aq) + NO3⁻(aq)',
        'HClO3': 'H⁺(aq) + ClO3⁻(aq)',
        'HClO4': 'H⁺(aq) + ClO4⁻(aq)',
        'H2SO4': '2H⁺(aq) + SO4²⁻(aq)',
        
        # Strong bases
        'NaOH': 'Na⁺(aq) + OH⁻(aq)',
        'KOH': 'K⁺(aq) + OH⁻(aq)',
        'LiOH': 'Li⁺(aq) + OH⁻(aq)',
        'Ca(OH)2': 'Ca²⁺(aq) + 2OH⁻(aq)',
        'Ba(OH)2': 'Ba²⁺(aq) + 2OH⁻(aq)',
        'Sr(OH)2': 'Sr²⁺(aq) + 2OH⁻(aq)',
        
        # Common salts
        'NaCl': 'Na⁺(aq) + Cl⁻(aq)',
        'KCl': 'K⁺(aq) + Cl⁻(aq)',
        'NaNO3': 'Na⁺(aq) + NO3⁻(aq)',
        'KNO3': 'K⁺(aq) + NO3⁻(aq)',
        'Na2SO4': '2Na⁺(aq) + SO4²⁻(aq)',
        'K2SO4': '2K⁺(aq) + SO4²⁻(aq)',
        'AgNO3': 'Ag⁺(aq) + NO3⁻(aq)',
        'NH4Cl': 'NH4⁺(aq) + Cl⁻(aq)',
        'MgCl2': 'Mg²⁺(aq) + 2Cl⁻(aq)',
        'CaCl2': 'Ca²⁺(aq) + 2Cl⁻(aq)',
        'BaCl2': 'Ba²⁺(aq) + 2Cl⁻(aq)',
        
        # Weak electrolytes stay molecular
        'CH3COOH': 'CH3COOH(aq)',
        'NH3': 'NH3(aq)',
        'H2CO3': 'H2CO3(aq)',
        
        # Common precipitates (stay solid)
        'AgCl': 'AgCl(s)',
        'BaSO4': 'BaSO4(s)',
        'CaCO3': 'CaCO3(s)',
        'PbI2': 'PbI2(s)',
        'Fe(OH)3': 'Fe(OH)3(s)',
        'Cu(OH)2': 'Cu(OH)2(s)',
        
        # Gases and liquids
        'H2O': 'H2O(l)',
        'CO2': 'CO2(g)',
        'H2': 'H2(g)',
        'O2': 'O2(g)',
        'N2': 'N2(g)',
        'Cl2': 'Cl2(g)',
        'NH3': 'NH3(g)',
    }
    
    def convert_compound(comp: ReactionComponent) -> str:
        """Convert a compound to its ionic form."""
        formula = comp.formula
        
        # Check if we have a predefined ionic form
        if formula in ionic_forms:
            ionic_form = ionic_forms[formula]
            # Apply coefficient to ionic form
            if comp.coefficient > 1:
                # Distribute coefficient to all ions
                parts = ionic_form.split(' + ')
                parts_with_coeff = []
                for part in parts:
                    # Extract any existing coefficient
                    match = re.match(r'^(\d*)(.+)$', part.strip())
                    if match:
                        existing_coeff_str, ion = match.groups()
                        existing_coeff = int(existing_coeff_str) if existing_coeff_str else 1
                        new_coeff = existing_coeff * comp.coefficient
                        parts_with_coeff.append(f"{new_coeff if new_coeff > 1 else ''}{ion}")
                ionic_form = ' + '.join(parts_with_coeff)
            return ionic_form
        
        # Default: keep original with state if available
        state = f"({comp.state})" if comp.state else "(aq)"
        coeff_str = str(comp.coefficient) if comp.coefficient > 1 else ''
        return f"{coeff_str}{formula}{state}"
    
    # Convert reactants and products
    ionic_reactants = [convert_compound(comp) for comp in parsed["reactants"]]
    ionic_products = [convert_compound(comp) for comp in parsed["products"]]
    
    separator = parsed.get("separator", "->")
    ionic_eq = " + ".join(ionic_reactants) + f" {separator} " + " + ".join(ionic_products)
    
    return ionic_eq


def net_ionic(ionic_eq: str) -> str:
    """Generate net ionic equation by removing spectator ions."""
    # Common spectator ions
    spectator_ions = [
        r'Na\+\s*\(aq\)', r'K\+\s*\(aq\)', r'Li\+\s*\(aq\)',
        r'NO3\-\s*\(aq\)', r'Cl\-\s*\(aq\)', r'Br\-\s*\(aq\)',
        r'I\-\s*\(aq\)', r'SO4\2-\s*\(aq\)'
    ]
    
    if '->' in ionic_eq:
        left, right = ionic_eq.split('->')
    elif '<->' in ionic_eq:
        left, right = ionic_eq.split('<->')
    else:
        return ionic_eq
    
    # Remove spectator ions that appear on both sides
    for ion_pattern in spectator_ions:
        ion_regex = re.compile(ion_pattern)
        
        # Find all occurrences on left and right
        left_matches = list(ion_regex.finditer(left))
        right_matches = list(ion_regex.finditer(right))
        
        if left_matches and right_matches:
            # Remove from both sides
            for match in reversed(left_matches):
                start, end = match.span()
                # Also remove preceding coefficient if any
                # Look back for coefficient
                coeff_match = re.search(r'(\d+)\s*$', left[:start])
                if coeff_match:
                    start = coeff_match.start()
                # Remove the ion and any surrounding +
                left = left[:start].rstrip('+ ') + left[end:].lstrip()
            
            for match in reversed(right_matches):
                start, end = match.span()
                coeff_match = re.search(r'(\d+)\s*$', right[:start])
                if coeff_match:
                    start = coeff_match.start()
                right = right[:start].rstrip('+ ') + right[end:].lstrip()
    
    # Clean up extra + signs and spaces
    left = re.sub(r'\s*\+\s*\+', '+', left).strip('+ ').strip()
    right = re.sub(r'\s*\+\s*\+', '+', right).strip('+ ').strip()
    
    # Remove empty parentheses
    left = left.replace('()', '').strip()
    right = right.replace('()', '').strip()
    
    # Get separator
    separator = '->' if '->' in ionic_eq else '<->'
    
    if left and right:
        return f"{left} {separator} {right}"
    
    return ionic_eq


def get_reaction_type(equation: str) -> str:
    """Determine the type of chemical reaction with enhanced detection."""
    equation_lower = equation.lower()
    
    # Parse the equation
    if '->' in equation:
        left, right = equation.split('->')
    elif '<->' in equation:
        left, right = equation.split('<->')
    else:
        return "General"
    
    left = left.strip()
    right = right.strip()
    
    # Parse compounds
    def split_compounds(side: str) -> List[str]:
        return [c.strip() for c in re.split(r'\s*\+\s*', side) if c.strip()]
    
    left_compounds = split_compounds(left)
    right_compounds = split_compounds(right)
    
    # Check for combustion
    if ('o2' in equation_lower or 'oxygen' in equation_lower) and \
       ('co2' in equation_lower and 'h2o' in equation_lower):
        return "Combustion"
    
    # Check for synthesis (A + B → C)
    if len(left_compounds) >= 2 and len(right_compounds) == 1:
        return "Synthesis"
    
    # Check for decomposition (A → B + C)
    if len(left_compounds) == 1 and len(right_compounds) >= 2:
        return "Decomposition"
    
    # Check for single replacement (A + BC → AC + B)
    # Look for pattern: element + compound → different element + different compound
    if len(left_compounds) == 2 and len(right_compounds) == 2:
        # Try to identify an element (no numbers in formula)
        elements = []
        compounds = []
        for comp in left_compounds:
            if not any(c.isdigit() for c in comp if c not in '+-'):
                elements.append(comp)
            else:
                compounds.append(comp)
        
        if len(elements) == 1 and len(compounds) == 1:
            return "Single Replacement"
    
    # Check for double replacement (AB + CD → AD + CB)
    if len(left_compounds) == 2 and len(right_compounds) == 2:
        # Look for precipitation indicators
        precipitates = ['agcl', 'baso4', 'caco3', 'pbi2', 'pbso4']
        if any(ppt in equation_lower for ppt in precipitates):
            return "Precipitation"
        
        # Look for acid-base indicators
        acids = ['hcl', 'h2so4', 'hno3', 'h3po4', 'ch3cooh']
        bases = ['naoh', 'koh', 'ca(oh)2', 'nh3', 'nh4oh']
        if any(acid in equation_lower for acid in acids) and \
           any(base in equation_lower for base in bases):
            return "Acid-Base Neutralization"
        
        return "Double Replacement"
    
    # Check for redox (look for oxidation state changes)
    redox_indicators = ['kmno4', 'k2cr2o7', 'h2o2', 'fe2+', 'fe3+', 'cu+', 'cu2+']
    if any(indicator in equation_lower for indicator in redox_indicators):
        return "Redox"
    
    # Check for gas evolution
    gases = ['co2', 'h2', 'o2', 'cl2', 'so2', 'nh3', 'h2s']
    if any(gas in equation_lower for gas in gases):
        return "Gas Evolution"
    
    return "General"


def get_reaction_type_enhanced(equation: str) -> Dict[str, Any]:
    """Enhanced reaction type detection with more details."""
    base_type = get_reaction_type(equation)
    
    equation_lower = equation.lower()
    
    analysis = {
        "type": base_type,
        "is_redox": False,
        "is_precipitation": False,
        "is_gas_forming": False,
        "is_acid_base": False,
        "is_combustion": False,
        "is_reversible": '<->' in equation or '⇌' in equation
    }
    
    # Set specific flags
    analysis["is_combustion"] = base_type == "Combustion"
    analysis["is_precipitation"] = base_type == "Precipitation"
    analysis["is_acid_base"] = "Neutralization" in base_type
    
    # Check for gas formation
    gases = ['co2', 'h2', 'o2', 'cl2', 'so2', 'nh3', 'h2s', 'n2']
    analysis["is_gas_forming"] = any(gas in equation_lower for gas in gases)
    
    # Check for redox
    redox_patterns = [
        r'[A-Za-z]\([I|II|III|IV|V|VI|VII]\)',  # Roman numerals
        r'[+-]\d+',  # Formal charges
        'oxid', 'reduc', 'electron'
    ]
    analysis["is_redox"] = any(
        re.search(pattern, equation_lower) for pattern in redox_patterns
    ) or base_type == "Redox"
    
    return analysis


@lru_cache(maxsize=128)
def calculate_molar_mass(formula: str) -> float:
    """Calculate molar mass using the enhanced formula parser."""
    try:
        elements = parse_chemical_formula(formula)
        if not elements:
            return 0.0
        
        # Atomic masses (in g/mol)
        atomic_masses = {
            'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122,
            'B': 10.81, 'C': 12.011, 'N': 14.007, 'O': 16.00,
            'F': 19.00, 'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305,
            'Al': 26.982, 'Si': 28.085, 'P': 30.974, 'S': 32.06,
            'Cl': 35.45, 'K': 39.098, 'Ca': 40.078, 'Sc': 44.956,
            'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938,
            'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546,
            'Zn': 65.38, 'Ga': 69.723, 'Ge': 72.63, 'As': 74.922,
            'Se': 78.96, 'Br': 79.904, 'Kr': 83.798, 'Rb': 85.468,
            'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224, 'Nb': 92.906,
            'Mo': 95.95, 'Tc': 98, 'Ru': 101.07, 'Rh': 102.91,
            'Pd': 106.42, 'Ag': 107.87, 'Cd': 112.41, 'In': 114.82,
            'Sn': 118.71, 'Sb': 121.76, 'Te': 127.60, 'I': 126.90,
            'Xe': 131.29, 'Cs': 132.91, 'Ba': 137.33, 'La': 138.91,
            'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24, 'Pm': 145,
            'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.93,
            'Dy': 162.50, 'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93,
            'Yb': 173.05, 'Lu': 174.97, 'Hf': 178.49, 'Ta': 180.95,
            'W': 183.84, 'Re': 186.21, 'Os': 190.23, 'Ir': 192.22,
            'Pt': 195.08, 'Au': 196.97, 'Hg': 200.59, 'Tl': 204.38,
            'Pb': 207.2, 'Bi': 208.98, 'Po': 209, 'At': 210,
            'Rn': 222, 'Fr': 223, 'Ra': 226, 'Ac': 227, 'Th': 232.04,
            'Pa': 231.04, 'U': 238.03, 'Np': 237, 'Pu': 244,
            'Am': 243, 'Cm': 247, 'Bk': 247, 'Cf': 251,
            'Es': 252, 'Fm': 257, 'Md': 258, 'No': 259,
            'Lr': 266, 'Rf': 267, 'Db': 268, 'Sg': 269,
            'Bh': 270, 'Hs': 270, 'Mt': 278, 'Ds': 281,
            'Rg': 282, 'Cn': 285, 'Nh': 286, 'Fl': 289,
            'Mc': 290, 'Lv': 293, 'Ts': 294, 'Og': 294
        }
        
        total_mass = 0.0
        for element, count in elements.items():
            mass = atomic_masses.get(element, 0.0)
            if mass == 0.0:
                # Estimate based on atomic number if not found
                # This is a rough estimate for demonstration
                print(f"Warning: Unknown element {element}, using estimated mass")
                mass = 30.0
            total_mass += mass * count
        
        return round(total_mass, 4)
        
    except Exception as e:
        print(f"Molar mass calculation error: {e}")
        return 0.0


def predict_products(reactants: str, reaction_type: str = "auto") -> str:
    """Predict products for given reactants with enhanced logic."""
    reactants_clean = reactants.strip().lower()
    
    # Comprehensive prediction database
    predictions = {
        # Acid-Base reactions
        "hcl + naoh": "HCl + NaOH -> NaCl + H2O",
        "h2so4 + naoh": "H2SO4 + 2NaOH -> Na2SO4 + 2H2O",
        "hno3 + koh": "HNO3 + KOH -> KNO3 + H2O",
        "ch3cooh + naoh": "CH3COOH + NaOH -> CH3COONa + H2O",
        "hcl + nh3": "HCl + NH3 -> NH4Cl",
        "h2so4 + nh3": "H2SO4 + 2NH3 -> (NH4)2SO4",
        
        # Combustion reactions
        "ch4 + o2": "CH4 + 2O2 -> CO2 + 2H2O",
        "c2h6 + o2": "2C2H6 + 7O2 -> 4CO2 + 6H2O",
        "c3h8 + o2": "C3H8 + 5O2 -> 3CO2 + 4H2O",
        "c6h12o6 + o2": "C6H12O6 + 6O2 -> 6CO2 + 6H2O",
        "c2h5oh + o2": "C2H5OH + 3O2 -> 2CO2 + 3H2O",
        
        # Single replacement
        "zn + hcl": "Zn + 2HCl -> ZnCl2 + H2",
        "fe + h2so4": "Fe + H2SO4 -> FeSO4 + H2",
        "cu + agno3": "Cu + 2AgNO3 -> Cu(NO3)2 + 2Ag",
        "mg + hcl": "Mg + 2HCl -> MgCl2 + H2",
        "al + hcl": "2Al + 6HCl -> 2AlCl3 + 3H2",
        "zn + cuso4": "Zn + CuSO4 -> ZnSO4 + Cu",
        
        # Double replacement/precipitation
        "agno3 + nacl": "AgNO3 + NaCl -> AgCl + NaNO3",
        "bacl2 + na2so4": "BaCl2 + Na2SO4 -> BaSO4 + 2NaCl",
        "pb(no3)2 + ki": "Pb(NO3)2 + 2KI -> PbI2 + 2KNO3",
        "cacl2 + na2co3": "CaCl2 + Na2CO3 -> CaCO3 + 2NaCl",
        "agno3 + kbr": "AgNO3 + KBr -> AgBr + KNO3",
        
        # Decomposition
        "caco3": "CaCO3 -> CaO + CO2",
        "h2o2": "2H2O2 -> 2H2O + O2",
        "kclo3": "2KClO3 -> 2KCl + 3O2",
        "nh4no3": "NH4NO3 -> N2O + 2H2O",
        "h2co3": "H2CO3 -> H2O + CO2",
        "h2so4": "H2SO4 -> H2O + SO3",
        
        # Synthesis
        "h2 + o2": "2H2 + O2 -> 2H2O",
        "na + cl2": "2Na + Cl2 -> 2NaCl",
        "mg + o2": "2Mg + O2 -> 2MgO",
        "ca + o2": "2Ca + O2 -> 2CaO",
        "so2 + o2": "2SO2 + O2 -> 2SO3",
        "n2 + h2": "N2 + 3H2 -> 2NH3",
        
        # Gas evolution
        "caco3 + hcl": "CaCO3 + 2HCl -> CaCl2 + CO2 + H2O",
        "na2co3 + hcl": "Na2CO3 + 2HCl -> 2NaCl + CO2 + H2O",
        "nahco3 + hcl": "NaHCO3 + HCl -> NaCl + CO2 + H2O",
        "zn + h2so4": "Zn + H2SO4 -> ZnSO4 + H2",
        "fe + h2so4": "Fe + H2SO4 -> FeSO4 + H2",
        
        # Redox
        "kmno4 + hcl": "2KMnO4 + 16HCl -> 2KCl + 2MnCl2 + 5Cl2 + 8H2O",
        "k2cr2o7 + hcl": "K2Cr2O7 + 14HCl -> 2KCl + 2CrCl3 + 3Cl2 + 7H2O",
        "cu + hno3": "3Cu + 8HNO3 -> 3Cu(NO3)2 + 2NO + 4H2O",
        
        # Complex formation
        "agcl + nh3": "AgCl + 2NH3 -> [Ag(NH3)2]Cl",
        "cu(oh)2 + nh3": "Cu(OH)2 + 4NH3 -> [Cu(NH3)4](OH)2",
    }
    
    # Try exact matches first
    for pattern, prediction in predictions.items():
        # Create a normalized version for comparison
        norm_pattern = re.sub(r'\s+', ' ', pattern.strip().lower())
        norm_reactants = re.sub(r'\s+', ' ', reactants_clean)
        
        if norm_pattern == norm_reactants:
            return prediction
    
    # Try partial matches
    for pattern, prediction in predictions.items():
        norm_pattern = re.sub(r'\s+', ' ', pattern.strip().lower())
        norm_reactants = re.sub(r'\s+', ' ', reactants_clean)
        
        # Check if all compounds in pattern are in reactants
        pattern_compounds = set(re.findall(r'[A-Za-z0-9\(\)]+', norm_pattern))
        reactant_compounds = set(re.findall(r'[A-Za-z0-9\(\)]+', norm_reactants))
        
        if pattern_compounds.issubset(reactant_compounds):
            return prediction
    
    # Fallback based on reaction type
    if reaction_type != "auto":
        if reaction_type.lower() in ["combustion", "burn"]:
            return f"{reactants} + O2 -> CO2 + H2O"
        elif reaction_type.lower() in ["acid-base", "neutralization"]:
            return f"{reactants} -> Salt + H2O"
        elif reaction_type.lower() in ["single replacement", "displacement"]:
            return f"{reactants} -> New compound + Element"
        elif reaction_type.lower() in ["double replacement", "metathesis"]:
            return f"{reactants} -> Two new compounds"
        elif reaction_type.lower() == "decomposition":
            return f"{reactants} -> Simpler compounds"
        elif reaction_type.lower() == "synthesis":
            return f"{reactants} -> Single compound"
    
    # General fallback
    return f"{reactants} -> Products"


def limiting_reagent(balanced_eq: str, reactants: List[str], amounts: List[float]) -> Tuple[float, str, Dict[str, float]]:
    """Calculate limiting reagent with stoichiometry."""
    try:
        if not reactants or not amounts or len(reactants) != len(amounts):
            return 0.0, "Invalid input", {}
        
        parsed = parse_reaction_string(balanced_eq)
        if not parsed:
            return 0.0, "Parse error", {}
        
        # Build stoichiometric ratios from balanced equation
        stoich = {}
        for comp in parsed["reactants"]:
            stoich[comp.formula] = comp.coefficient
        
        # Convert all amounts to moles
        moles = {}
        molar_masses = {}
        
        for r, amt in zip(reactants, amounts):
            mm = calculate_molar_mass(r)
            molar_masses[r] = mm
            if mm > 0:
                moles[r] = amt / mm
            else:
                moles[r] = 0.0
        
        # Calculate mole ratio available vs required
        ratios = {}
        for r in reactants:
            if r in stoich and stoich[r] > 0:
                ratios[r] = moles[r] / stoich[r]
            else:
                ratios[r] = float('inf')
        
        if not ratios or all(v == float('inf') for v in ratios.values()):
            return 0.0, "No valid ratios", {}
        
        # Find limiting reagent (smallest ratio)
        limiting = min(ratios, key=ratios.get)
        limiting_moles = moles[limiting]
        limiting_ratio = ratios[limiting]
        
        # Calculate excess amounts
        excess = {}
        for r in reactants:
            if r != limiting and r in stoich and stoich[r] > 0:
                required_moles = limiting_ratio * stoich[r]
                available_moles = moles[r]
                excess_moles = available_moles - required_moles
                if excess_moles > 0:
                    excess[r] = {
                        'moles': excess_moles,
                        'mass': excess_moles * molar_masses[r]
                    }
        
        return limiting_moles, limiting, excess
        
    except Exception as e:
        print(f"Limiting reagent error: {e}")
        return 0.0, "Error", {}


def theoretical_yield(balanced_eq: str, product: str, limiting_amount: float, limiting_reagent: str) -> float:
    """Calculate theoretical yield of a product."""
    try:
        parsed = parse_reaction_string(balanced_eq)
        if not parsed:
            return 0.0
        
        # Find stoichiometric coefficients
        product_stoich = 0
        for comp in parsed["products"]:
            if comp.formula == product:
                product_stoich = comp.coefficient
                break
        
        if product_stoich == 0:
            return 0.0  # Product not found in equation
        
        limiting_stoich = 0
        for comp in parsed["reactants"]:
            if comp.formula == limiting_reagent:
                limiting_stoich = comp.coefficient
                break
        
        if limiting_stoich == 0:
            return 0.0  # Limiting reagent not found in equation
        
        # Calculate molar masses
        mm_limiting = calculate_molar_mass(limiting_reagent)
        mm_product = calculate_molar_mass(product)
        
        if mm_limiting == 0 or mm_product == 0:
            return 0.0
        
        # Convert limiting amount to moles
        moles_limiting = limiting_amount / mm_limiting
        
        # Calculate moles of product using stoichiometric ratio
        moles_product = moles_limiting * (product_stoich / limiting_stoich)
        
        # Convert to mass
        yield_amount = moles_product * mm_product
        
        return round(yield_amount, 4)
        
    except Exception as e:
        print(f"Theoretical yield error: {e}")
        return 0.0


def percent_yield(actual: float, theoretical: float) -> float:
    """Calculate percent yield."""
    if theoretical == 0:
        return 0.0
    return round((actual / theoretical) * 100, 2)


def calculate_enthalpy_change(balanced_eq: str) -> float:
    """Estimate standard enthalpy change for reaction."""
    try:
        parsed = parse_reaction_string(balanced_eq)
        if not parsed:
            return 0.0
        
        # Standard formation enthalpies (kJ/mol)
        formation_enthalpies = {
            'H2O': -285.8, 'CO2': -393.5, 'NaCl': -411.2, 'HCl': -92.3,
            'NaOH': -425.6, 'H2SO4': -814.0, 'NH3': -45.9, 'CH4': -74.6,
            'O2': 0, 'H2': 0, 'N2': 0, 'C': 0, 'Fe2O3': -824.2,
            'CaCO3': -1207, 'CaO': -635, 'SO2': -296.8, 'NO2': 33.2,
            'NH4Cl': -314.4, 'AgCl': -127.0, 'BaSO4': -1473, 'PbI2': -176,
            'CH3COOH': -484.5, 'C2H5OH': -277.7, 'C6H12O6': -1274,
            'Fe': 0, 'Cu': 0, 'Ag': 0, 'Zn': 0, 'Al': 0,
            'KCl': -436.5, 'KOH': -424.8, 'Na2SO4': -1387,
            'MgO': -601.6, 'Al2O3': -1676, 'SiO2': -910.9
        }
        
        # Calculate ΔH = ΣΔHf(products) - ΣΔHf(reactants)
        delta_h = 0.0
        
        # Products
        for comp in parsed["products"]:
            enthalpy = formation_enthalpies.get(comp.formula, 0.0)
            delta_h += enthalpy * comp.coefficient
        
        # Reactants
        for comp in parsed["reactants"]:
            enthalpy = formation_enthalpies.get(comp.formula, 0.0)
            delta_h -= enthalpy * comp.coefficient
        
        return round(delta_h, 2)
        
    except Exception as e:
        print(f"Enthalpy calculation error: {e}")
        return 0.0


def predict_precipitate(equation: str) -> List[str]:
    """Predict precipitates in reaction."""
    precipitates = []
    
    # Common insoluble compounds (precipitates)
    insoluble_compounds = [
        'AgCl', 'AgBr', 'AgI', 'Ag2SO4',
        'BaSO4', 'BaCO3', 'Ba3(PO4)2',
        'CaCO3', 'CaSO4', 'Ca3(PO4)2',
        'PbCl2', 'PbI2', 'PbSO4', 'PbCO3',
        'Fe(OH)3', 'Fe(OH)2', 'FeS',
        'Cu(OH)2', 'CuS', 'CuCO3',
        'Zn(OH)2', 'ZnS',
        'Mg(OH)2', 'MgCO3',
        'Al(OH)3',
        'Hg2Cl2', 'HgS'
    ]
    
    for compound in insoluble_compounds:
        if compound in equation:
            precipitates.append(compound)
    
    return precipitates


def predict_gas_formation(equation: str) -> List[str]:
    """Predict gases formed in reaction."""
    gases = []
    
    common_gases = [
        'CO2', 'H2', 'O2', 'N2', 'Cl2', 'F2',
        'NH3', 'SO2', 'SO3', 'NO', 'NO2', 'N2O',
        'HCl', 'HBr', 'HI', 'H2S', 'PH3',
        'CH4', 'C2H6', 'C3H8', 'C2H4', 'C2H2'
    ]
    
    for gas in common_gases:
        if gas in equation:
            gases.append(gas)
    
    return gases


def calculate_mass_from_moles(moles: float, formula: str) -> float:
    """Calculate mass from moles for a given formula."""
    molar_mass = calculate_molar_mass(formula)
    if molar_mass == 0:
        return 0.0
    return round(moles * molar_mass, 4)


def calculate_moles_from_mass(mass: float, formula: str) -> float:
    """Calculate moles from mass for a given formula."""
    molar_mass = calculate_molar_mass(formula)
    if molar_mass == 0:
        return 0.0
    return round(mass / molar_mass, 4)


def detect_reaction_type(equation: str) -> str:
    """Detect reaction type from equation string."""
    return get_reaction_type(equation)


def init_elements_data():
    """Initialize periodic table data with all 118 elements."""
    elements = [
        {"symbol": "H", "name": "Hydrogen", "atomic_number": 1, "atomic_mass": 1.008, "group": 1, "period": 1, "category": "nonmetal"},
        {"symbol": "He", "name": "Helium", "atomic_number": 2, "atomic_mass": 4.0026, "group": 18, "period": 1, "category": "noble"},
        {"symbol": "Li", "name": "Lithium", "atomic_number": 3, "atomic_mass": 6.94, "group": 1, "period": 2, "category": "alkali"},
        {"symbol": "Be", "name": "Beryllium", "atomic_number": 4, "atomic_mass": 9.0122, "group": 2, "period": 2, "category": "alkaline"},
        {"symbol": "B", "name": "Boron", "atomic_number": 5, "atomic_mass": 10.81, "group": 13, "period": 2, "category": "semimetal"},
        {"symbol": "C", "name": "Carbon", "atomic_number": 6, "atomic_mass": 12.011, "group": 14, "period": 2, "category": "nonmetal"},
        {"symbol": "N", "name": "Nitrogen", "atomic_number": 7, "atomic_mass": 14.007, "group": 15, "period": 2, "category": "nonmetal"},
        {"symbol": "O", "name": "Oxygen", "atomic_number": 8, "atomic_mass": 16.00, "group": 16, "period": 2, "category": "nonmetal"},
        {"symbol": "F", "name": "Fluorine", "atomic_number": 9, "atomic_mass": 19.00, "group": 17, "period": 2, "category": "halogen"},
        {"symbol": "Ne", "name": "Neon", "atomic_number": 10, "atomic_mass": 20.180, "group": 18, "period": 2, "category": "noble"},
        {"symbol": "Na", "name": "Sodium", "atomic_number": 11, "atomic_mass": 22.990, "group": 1, "period": 3, "category": "alkali"},
        {"symbol": "Mg", "name": "Magnesium", "atomic_number": 12, "atomic_mass": 24.305, "group": 2, "period": 3, "category": "alkaline"},
        {"symbol": "Al", "name": "Aluminum", "atomic_number": 13, "atomic_mass": 26.982, "group": 13, "period": 3, "category": "basic"},
        {"symbol": "Si", "name": "Silicon", "atomic_number": 14, "atomic_mass": 28.085, "group": 14, "period": 3, "category": "semimetal"},
        {"symbol": "P", "name": "Phosphorus", "atomic_number": 15, "atomic_mass": 30.974, "group": 15, "period": 3, "category": "nonmetal"},
        {"symbol": "S", "name": "Sulfur", "atomic_number": 16, "atomic_mass": 32.06, "group": 16, "period": 3, "category": "nonmetal"},
        {"symbol": "Cl", "name": "Chlorine", "atomic_number": 17, "atomic_mass": 35.45, "group": 17, "period": 3, "category": "halogen"},
        {"symbol": "Ar", "name": "Argon", "atomic_number": 18, "atomic_mass": 39.948, "group": 18, "period": 3, "category": "noble"},
        {"symbol": "K", "name": "Potassium", "atomic_number": 19, "atomic_mass": 39.098, "group": 1, "period": 4, "category": "alkali"},
        {"symbol": "Ca", "name": "Calcium", "atomic_number": 20, "atomic_mass": 40.078, "group": 2, "period": 4, "category": "alkaline"},
        {"symbol": "Sc", "name": "Scandium", "atomic_number": 21, "atomic_mass": 44.956, "group": 3, "period": 4, "category": "transition"},
        {"symbol": "Ti", "name": "Titanium", "atomic_number": 22, "atomic_mass": 47.867, "group": 4, "period": 4, "category": "transition"},
        {"symbol": "V", "name": "Vanadium", "atomic_number": 23, "atomic_mass": 50.942, "group": 5, "period": 4, "category": "transition"},
        {"symbol": "Cr", "name": "Chromium", "atomic_number": 24, "atomic_mass": 51.996, "group": 6, "period": 4, "category": "transition"},
        {"symbol": "Mn", "name": "Manganese", "atomic_number": 25, "atomic_mass": 54.938, "group": 7, "period": 4, "category": "transition"},
        {"symbol": "Fe", "name": "Iron", "atomic_number": 26, "atomic_mass": 55.845, "group": 8, "period": 4, "category": "transition"},
        {"symbol": "Co", "name": "Cobalt", "atomic_number": 27, "atomic_mass": 58.933, "group": 9, "period": 4, "category": "transition"},
        {"symbol": "Ni", "name": "Nickel", "atomic_number": 28, "atomic_mass": 58.693, "group": 10, "period": 4, "category": "transition"},
        {"symbol": "Cu", "name": "Copper", "atomic_number": 29, "atomic_mass": 63.546, "group": 11, "period": 4, "category": "transition"},
        {"symbol": "Zn", "name": "Zinc", "atomic_number": 30, "atomic_mass": 65.38, "group": 12, "period": 4, "category": "transition"},
        {"symbol": "Ga", "name": "Gallium", "atomic_number": 31, "atomic_mass": 69.723, "group": 13, "period": 4, "category": "basic"},
        {"symbol": "Ge", "name": "Germanium", "atomic_number": 32, "atomic_mass": 72.63, "group": 14, "period": 4, "category": "semimetal"},
        {"symbol": "As", "name": "Arsenic", "atomic_number": 33, "atomic_mass": 74.922, "group": 15, "period": 4, "category": "semimetal"},
        {"symbol": "Se", "name": "Selenium", "atomic_number": 34, "atomic_mass": 78.96, "group": 16, "period": 4, "category": "nonmetal"},
        {"symbol": "Br", "name": "Bromine", "atomic_number": 35, "atomic_mass": 79.904, "group": 17, "period": 4, "category": "halogen"},
        {"symbol": "Kr", "name": "Krypton", "atomic_number": 36, "atomic_mass": 83.798, "group": 18, "period": 4, "category": "noble"},
        {"symbol": "Rb", "name": "Rubidium", "atomic_number": 37, "atomic_mass": 85.468, "group": 1, "period": 5, "category": "alkali"},
        {"symbol": "Sr", "name": "Strontium", "atomic_number": 38, "atomic_mass": 87.62, "group": 2, "period": 5, "category": "alkaline"},
        {"symbol": "Y", "name": "Yttrium", "atomic_number": 39, "atomic_mass": 88.906, "group": 3, "period": 5, "category": "transition"},
        {"symbol": "Zr", "name": "Zirconium", "atomic_number": 40, "atomic_mass": 91.224, "group": 4, "period": 5, "category": "transition"},
        {"symbol": "Nb", "name": "Niobium", "atomic_number": 41, "atomic_mass": 92.906, "group": 5, "period": 5, "category": "transition"},
        {"symbol": "Mo", "name": "Molybdenum", "atomic_number": 42, "atomic_mass": 95.95, "group": 6, "period": 5, "category": "transition"},
        {"symbol": "Tc", "name": "Technetium", "atomic_number": 43, "atomic_mass": 98, "group": 7, "period": 5, "category": "transition"},
        {"symbol": "Ru", "name": "Ruthenium", "atomic_number": 44, "atomic_mass": 101.07, "group": 8, "period": 5, "category": "transition"},
        {"symbol": "Rh", "name": "Rhodium", "atomic_number": 45, "atomic_mass": 102.91, "group": 9, "period": 5, "category": "transition"},
        {"symbol": "Pd", "name": "Palladium", "atomic_number": 46, "atomic_mass": 106.42, "group": 10, "period": 5, "category": "transition"},
        {"symbol": "Ag", "name": "Silver", "atomic_number": 47, "atomic_mass": 107.87, "group": 11, "period": 5, "category": "transition"},
        {"symbol": "Cd", "name": "Cadmium", "atomic_number": 48, "atomic_mass": 112.41, "group": 12, "period": 5, "category": "transition"},
        {"symbol": "In", "name": "Indium", "atomic_number": 49, "atomic_mass": 114.82, "group": 13, "period": 5, "category": "basic"},
        {"symbol": "Sn", "name": "Tin", "atomic_number": 50, "atomic_mass": 118.71, "group": 14, "period": 5, "category": "basic"},
        {"symbol": "Sb", "name": "Antimony", "atomic_number": 51, "atomic_mass": 121.76, "group": 15, "period": 5, "category": "semimetal"},
        {"symbol": "Te", "name": "Tellurium", "atomic_number": 52, "atomic_mass": 127.60, "group": 16, "period": 5, "category": "semimetal"},
        {"symbol": "I", "name": "Iodine", "atomic_number": 53, "atomic_mass": 126.90, "group": 17, "period": 5, "category": "halogen"},
        {"symbol": "Xe", "name": "Xenon", "atomic_number": 54, "atomic_mass": 131.29, "group": 18, "period": 5, "category": "noble"},
        {"symbol": "Cs", "name": "Cesium", "atomic_number": 55, "atomic_mass": 132.91, "group": 1, "period": 6, "category": "alkali"},
        {"symbol": "Ba", "name": "Barium", "atomic_number": 56, "atomic_mass": 137.33, "group": 2, "period": 6, "category": "alkaline"},
        {"symbol": "La", "name": "Lanthanum", "atomic_number": 57, "atomic_mass": 138.91, "group": 3, "period": 6, "category": "lanthanide"},
        {"symbol": "Ce", "name": "Cerium", "atomic_number": 58, "atomic_mass": 140.12, "group": 3, "period": 6, "category": "lanthanide"},
        {"symbol": "Pr", "name": "Praseodymium", "atomic_number": 59, "atomic_mass": 140.91, "group": 3, "period": 6, "category": "lanthanide"},
        {"symbol": "Nd", "name": "Neodymium", "atomic_number": 60, "atomic_mass": 144.24, "group": 3, "period": 6, "category": "lanthanide"},
        {"symbol": "Pm", "name": "Promethium", "atomic_number": 61, "atomic_mass": 145, "group": 3, "period": 6, "category": "lanthanide"},
        {"symbol": "Sm", "name": "Samarium", "atomic_number": 62, "atomic_mass": 150.36, "group": 3, "period": 6, "category": "lanthanide"},
        {"symbol": "Eu", "name": "Europium", "atomic_number": 63, "atomic_mass": 151.96, "group": 3, "period": 6, "category": "lanthanide"},
        {"symbol": "Gd", "name": "Gadolinium", "atomic_number": 64, "atomic_mass": 157.25, "group": 3, "period": 6, "category": "lanthanide"},
        {"symbol": "Tb", "name": "Terbium", "atomic_number": 65, "atomic_mass": 158.93, "group": 3, "period": 6, "category": "lanthanide"},
        {"symbol": "Dy", "name": "Dysprosium", "atomic_number": 66, "atomic_mass": 162.50, "group": 3, "period": 6, "category": "lanthanide"},
        {"symbol": "Ho", "name": "Holmium", "atomic_number": 67, "atomic_mass": 164.93, "group": 3, "period": 6, "category": "lanthanide"},
        {"symbol": "Er", "name": "Erbium", "atomic_number": 68, "atomic_mass": 167.26, "group": 3, "period": 6, "category": "lanthanide"},
        {"symbol": "Tm", "name": "Thulium", "atomic_number": 69, "atomic_mass": 168.93, "group": 3, "period": 6, "category": "lanthanide"},
        {"symbol": "Yb", "name": "Ytterbium", "atomic_number": 70, "atomic_mass": 173.05, "group": 3, "period": 6, "category": "lanthanide"},
        {"symbol": "Lu", "name": "Lutetium", "atomic_number": 71, "atomic_mass": 174.97, "group": 3, "period": 6, "category": "lanthanide"},
        {"symbol": "Hf", "name": "Hafnium", "atomic_number": 72, "atomic_mass": 178.49, "group": 4, "period": 6, "category": "transition"},
        {"symbol": "Ta", "name": "Tantalum", "atomic_number": 73, "atomic_mass": 180.95, "group": 5, "period": 6, "category": "transition"},
        {"symbol": "W", "name": "Tungsten", "atomic_number": 74, "atomic_mass": 183.84, "group": 6, "period": 6, "category": "transition"},
        {"symbol": "Re", "name": "Rhenium", "atomic_number": 75, "atomic_mass": 186.21, "group": 7, "period": 6, "category": "transition"},
        {"symbol": "Os", "name": "Osmium", "atomic_number": 76, "atomic_mass": 190.23, "group": 8, "period": 6, "category": "transition"},
        {"symbol": "Ir", "name": "Iridium", "atomic_number": 77, "atomic_mass": 192.22, "group": 9, "period": 6, "category": "transition"},
        {"symbol": "Pt", "name": "Platinum", "atomic_number": 78, "atomic_mass": 195.08, "group": 10, "period": 6, "category": "transition"},
        {"symbol": "Au", "name": "Gold", "atomic_number": 79, "atomic_mass": 196.97, "group": 11, "period": 6, "category": "transition"},
        {"symbol": "Hg", "name": "Mercury", "atomic_number": 80, "atomic_mass": 200.59, "group": 12, "period": 6, "category": "transition"},
        {"symbol": "Tl", "name": "Thallium", "atomic_number": 81, "atomic_mass": 204.38, "group": 13, "period": 6, "category": "basic"},
        {"symbol": "Pb", "name": "Lead", "atomic_number": 82, "atomic_mass": 207.2, "group": 14, "period": 6, "category": "basic"},
        {"symbol": "Bi", "name": "Bismuth", "atomic_number": 83, "atomic_mass": 208.98, "group": 15, "period": 6, "category": "basic"},
        {"symbol": "Po", "name": "Polonium", "atomic_number": 84, "atomic_mass": 209, "group": 16, "period": 6, "category": "semimetal"},
        {"symbol": "At", "name": "Astatine", "atomic_number": 85, "atomic_mass": 210, "group": 17, "period": 6, "category": "halogen"},
        {"symbol": "Rn", "name": "Radon", "atomic_number": 86, "atomic_mass": 222, "group": 18, "period": 6, "category": "noble"},
        {"symbol": "Fr", "name": "Francium", "atomic_number": 87, "atomic_mass": 223, "group": 1, "period": 7, "category": "alkali"},
        {"symbol": "Ra", "name": "Radium", "atomic_number": 88, "atomic_mass": 226, "group": 2, "period": 7, "category": "alkaline"},
        {"symbol": "Ac", "name": "Actinium", "atomic_number": 89, "atomic_mass": 227, "group": 3, "period": 7, "category": "actinide"},
        {"symbol": "Th", "name": "Thorium", "atomic_number": 90, "atomic_mass": 232.04, "group": 3, "period": 7, "category": "actinide"},
        {"symbol": "Pa", "name": "Protactinium", "atomic_number": 91, "atomic_mass": 231.04, "group": 3, "period": 7, "category": "actinide"},
        {"symbol": "U", "name": "Uranium", "atomic_number": 92, "atomic_mass": 238.03, "group": 3, "period": 7, "category": "actinide"},
        {"symbol": "Np", "name": "Neptunium", "atomic_number": 93, "atomic_mass": 237, "group": 3, "period": 7, "category": "actinide"},
        {"symbol": "Pu", "name": "Plutonium", "atomic_number": 94, "atomic_mass": 244, "group": 3, "period": 7, "category": "actinide"},
        {"symbol": "Am", "name": "Americium", "atomic_number": 95, "atomic_mass": 243, "group": 3, "period": 7, "category": "actinide"},
        {"symbol": "Cm", "name": "Curium", "atomic_number": 96, "atomic_mass": 247, "group": 3, "period": 7, "category": "actinide"},
        {"symbol": "Bk", "name": "Berkelium", "atomic_number": 97, "atomic_mass": 247, "group": 3, "period": 7, "category": "actinide"},
        {"symbol": "Cf", "name": "Californium", "atomic_number": 98, "atomic_mass": 251, "group": 3, "period": 7, "category": "actinide"},
        {"symbol": "Es", "name": "Einsteinium", "atomic_number": 99, "atomic_mass": 252, "group": 3, "period": 7, "category": "actinide"},
        {"symbol": "Fm", "name": "Fermium", "atomic_number": 100, "atomic_mass": 257, "group": 3, "period": 7, "category": "actinide"},
        {"symbol": "Md", "name": "Mendelevium", "atomic_number": 101, "atomic_mass": 258, "group": 3, "period": 7, "category": "actinide"},
        {"symbol": "No", "name": "Nobelium", "atomic_number": 102, "atomic_mass": 259, "group": 3, "period": 7, "category": "actinide"},
        {"symbol": "Lr", "name": "Lawrencium", "atomic_number": 103, "atomic_mass": 266, "group": 3, "period": 7, "category": "actinide"},
        {"symbol": "Rf", "name": "Rutherfordium", "atomic_number": 104, "atomic_mass": 267, "group": 4, "period": 7, "category": "transition"},
        {"symbol": "Db", "name": "Dubnium", "atomic_number": 105, "atomic_mass": 268, "group": 5, "period": 7, "category": "transition"},
        {"symbol": "Sg", "name": "Seaborgium", "atomic_number": 106, "atomic_mass": 269, "group": 6, "period": 7, "category": "transition"},
        {"symbol": "Bh", "name": "Bohrium", "atomic_number": 107, "atomic_mass": 270, "group": 7, "period": 7, "category": "transition"},
        {"symbol": "Hs", "name": "Hassium", "atomic_number": 108, "atomic_mass": 270, "group": 8, "period": 7, "category": "transition"},
        {"symbol": "Mt", "name": "Meitnerium", "atomic_number": 109, "atomic_mass": 278, "group": 9, "period": 7, "category": "transition"},
        {"symbol": "Ds", "name": "Darmstadtium", "atomic_number": 110, "atomic_mass": 281, "group": 10, "period": 7, "category": "transition"},
        {"symbol": "Rg", "name": "Roentgenium", "atomic_number": 111, "atomic_mass": 282, "group": 11, "period": 7, "category": "transition"},
        {"symbol": "Cn", "name": "Copernicium", "atomic_number": 112, "atomic_mass": 285, "group": 12, "period": 7, "category": "transition"},
        {"symbol": "Nh", "name": "Nihonium", "atomic_number": 113, "atomic_mass": 286, "group": 13, "period": 7, "category": "basic"},
        {"symbol": "Fl", "name": "Flerovium", "atomic_number": 114, "atomic_mass": 289, "group": 14, "period": 7, "category": "basic"},
        {"symbol": "Mc", "name": "Moscovium", "atomic_number": 115, "atomic_mass": 290, "group": 15, "period": 7, "category": "basic"},
        {"symbol": "Lv", "name": "Livermorium", "atomic_number": 116, "atomic_mass": 293, "group": 16, "period": 7, "category": "basic"},
        {"symbol": "Ts", "name": "Tennessine", "atomic_number": 117, "atomic_mass": 294, "group": 17, "period": 7, "category": "halogen"},
        {"symbol": "Og", "name": "Oganesson", "atomic_number": 118, "atomic_mass": 294, "group": 18, "period": 7, "category": "noble"}
    ]
    
    os.makedirs('data', exist_ok=True)
    with open('data/elements.json', 'w') as f:
        json.dump(elements, f, indent=2)
    
    print(f"✓ Initialized {len(elements)} elements")
    return elements


def test_balancing():
    """Test the balancing algorithm with known equations."""
    test_cases = [
        ("H2 + O2 -> H2O", "2H2 + O2 -> 2H2O"),
        ("CH4 + O2 -> CO2 + H2O", "CH4 + 2O2 -> CO2 + 2H2O"),
        ("Fe + O2 -> Fe2O3", "4Fe + 3O2 -> 2Fe2O3"),
        ("Al + HCl -> AlCl3 + H2", "2Al + 6HCl -> 2AlCl3 + 3H2"),
        ("Ca(OH)2 + H3PO4 -> Ca3(PO4)2 + H2O", "3Ca(OH)2 + 2H3PO4 -> Ca3(PO4)2 + 6H2O"),
        ("KMnO4 + HCl -> KCl + MnCl2 + Cl2 + H2O", "2KMnO4 + 16HCl -> 2KCl + 2MnCl2 + 5Cl2 + 8H2O"),
        ("C6H12O6 + O2 -> CO2 + H2O", "C6H12O6 + 6O2 -> 6CO2 + 6H2O"),
        ("NH3 + O2 -> NO + H2O", "4NH3 + 5O2 -> 4NO + 6H2O"),
        ("Na + Cl2 -> NaCl", "2Na + Cl2 -> 2NaCl"),
        ("Mg + N2 -> Mg3N2", "3Mg + N2 -> Mg3N2"),
    ]
    
    print("Testing equation balancing...")
    passed = 0
    total = len(test_cases)
    
    for input_eq, expected in test_cases:
        try:
            balanced, _ = balance_equation(input_eq)
            if balanced == expected:
                print(f"✓ PASS: {input_eq} -> {balanced}")
                passed += 1
            else:
                print(f"✗ FAIL: {input_eq}")
                print(f"  Expected: {expected}")
                print(f"  Got:      {balanced}")
        except Exception as e:
            print(f"✗ ERROR: {input_eq}")
            print(f"  Error: {e}")
    
    print(f"\nResults: {passed}/{total} passed ({passed/total*100:.1f}%)")
    return passed == total


def test_formula_parsing():
    """Test the formula parser."""
    test_formulas = [
        ("H2O", {"H": 2, "O": 1}),
        ("CO2", {"C": 1, "O": 2}),
        ("Ca(OH)2", {"Ca": 1, "O": 2, "H": 2}),
        ("Al2(SO4)3", {"Al": 2, "S": 3, "O": 12}),
        ("CuSO4·5H2O", {"Cu": 1, "S": 1, "O": 9, "H": 10}),
        ("K4[Fe(CN)6]", {"K": 4, "Fe": 1, "C": 6, "N": 6}),
        ("CH3COOH", {"C": 2, "H": 4, "O": 2}),
        ("C6H12O6", {"C": 6, "H": 12, "O": 6}),
    ]
    
    print("\nTesting formula parsing...")
    passed = 0
    
    for formula, expected in test_formulas:
        result = parse_chemical_formula(formula)
        if result == expected:
            print(f"✓ {formula}: {result}")
            passed += 1
        else:
            print(f"✗ {formula}")
            print(f"  Expected: {expected}")
            print(f"  Got:      {result}")
    
    print(f"Results: {passed}/{len(test_formulas)} passed")
    return passed == len(test_formulas)


def test_molar_mass():
    """Test molar mass calculations."""
    test_cases = [
        ("H2O", 18.015),
        ("CO2", 44.009),
        ("NaCl", 58.443),
        ("CH4", 16.043),
    ]
    
    print("\nTesting molar mass calculations...")
    for formula, expected in test_cases:
        result = calculate_molar_mass(formula)
        diff = abs(result - expected)
        if diff < 0.1:
            print(f"✓ {formula}: {result:.3f} g/mol")
        else:
            print(f"✗ {formula}: {result:.3f} g/mol (expected {expected:.3f})")


def main():
    """Main test function."""
    print("=" * 60)
    print("CHEMICAL EQUATION BALANCER - TEST SUITE")
    print("=" * 60)
    
    # Initialize elements database
    init_elements_data()
    
    # Run tests
    test_formula_parsing()
    test_balancing()
    test_molar_mass()
    
    # Demo
    print("\n" + "=" * 60)
    print("DEMONSTRATION")
    print("=" * 60)
    
    equations = [
        "H2 + O2 -> H2O",
        "CH4 + O2 -> CO2 + H2O",
        "Al + HCl -> AlCl3 + H2",
        "KMnO4 + HCl -> KCl + MnCl2 + Cl2 + H2O",
    ]
    
    for eq in equations:
        print(f"\nOriginal: {eq}")
        
        try:
            # Validate
            is_valid, message = validate_chemical_equation(eq)
            print(f"Validation: {message}")
            
            if is_valid:
                # Balance
                balanced, coeffs = validate_and_balance(eq)
                print(f"Balanced: {balanced}")
                print(f"Coefficients: {coeffs}")
                
                # Get reaction type
                rxn_type = get_reaction_type(balanced)
                print(f"Type: {rxn_type}")
                
                # Get enhanced analysis
                analysis = get_reaction_type_enhanced(balanced)
                print(f"Analysis: {analysis}")
                
                # Ionic equation
                ionic_eq = ionic_equation(balanced)
                print(f"Ionic: {ionic_eq}")
                
                # Net ionic
                net_eq = net_ionic(ionic_eq)
                print(f"Net ionic: {net_eq}")
                
                # Molar mass example
                if 'H2O' in balanced:
                    mm = calculate_molar_mass('H2O')
                    print(f"Molar mass of H2O: {mm} g/mol")
                
                # Enthalpy change
                delta_h = calculate_enthalpy_change(balanced)
                print(f"ΔH° ≈ {delta_h} kJ/mol")
                
        except ChemicalEquationError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
    
    print("\n" + "=" * 60)
    print("TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()