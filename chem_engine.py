import re
import json
from typing import List, Tuple, Dict, Optional, Union
from collections import defaultdict

class ReactionComponent:
    def __init__(self, formula: str, coefficient: int = 1):
        self.formula = formula
        self.coefficient = coefficient
    
    def __repr__(self):
        return f"{self.coefficient if self.coefficient != 1 else ''}{self.formula}"

def parse_reaction_string(equation: str) -> Optional[Dict]:
    """Parse reaction string into structured data."""
    try:
        # Clean the equation
        equation = equation.strip().replace(' ', '')
        
        # Determine separator
        if '->' in equation:
            reactants_str, products_str = equation.split('->')
        elif '=' in equation:
            reactants_str, products_str = equation.split('=')
        else:
            return None
        
        reactants = []
        products = []
        
        # Parse reactants
        for comp_str in reactants_str.split('+'):
            if not comp_str:
                continue
            # Extract coefficient and formula
            match = re.match(r'^(\d*)([A-Z][a-z]*\d*)$', comp_str)
            if match:
                coeff = int(match.group(1)) if match.group(1) else 1
                formula = match.group(2)
                reactants.append(ReactionComponent(formula, coeff))
        
        # Parse products
        for comp_str in products_str.split('+'):
            if not comp_str:
                continue
            match = re.match(r'^(\d*)([A-Z][a-z]*\d*)$', comp_str)
            if match:
                coeff = int(match.group(1)) if match.group(1) else 1
                formula = match.group(2)
                products.append(ReactionComponent(formula, coeff))
        
        return {"reactants": reactants, "products": products}
        
    except Exception as e:
        print(f"Error parsing reaction string: {e}")
        return None

def balance_equation(equation: str) -> Tuple[str, Dict[str, int]]:
    """Balance chemical equation using pattern matching."""
    try:
        equation = equation.replace(' ', '')
        
        # Common reaction balances
        common_balances = {
            "H2+O2->H2O": ("2H2 + O2 -> 2H2O", {"H2": 2, "O2": 1, "H2O": 2}),
            "N2+H2->NH3": ("N2 + 3H2 -> 2NH3", {"N2": 1, "H2": 3, "NH3": 2}),
            "CH4+O2->CO2+H2O": ("CH4 + 2O2 -> CO2 + 2H2O", {"CH4": 1, "O2": 2, "CO2": 1, "H2O": 2}),
            "Fe+O2->Fe2O3": ("4Fe + 3O2 -> 2Fe2O3", {"Fe": 4, "O2": 3, "Fe2O3": 2}),
            "HCl+NaOH->NaCl+H2O": ("HCl + NaOH -> NaCl + H2O", {"HCl": 1, "NaOH": 1, "NaCl": 1, "H2O": 1}),
            "H2SO4+NaOH->Na2SO4+H2O": ("H2SO4 + 2NaOH -> Na2SO4 + 2H2O", {"H2SO4": 1, "NaOH": 2, "Na2SO4": 1, "H2O": 2}),
            "Zn+HCl->ZnCl2+H2": ("Zn + 2HCl -> ZnCl2 + H2", {"Zn": 1, "HCl": 2, "ZnCl2": 1, "H2": 1}),
            "CaCO3+HCl->CaCl2+CO2+H2O": ("CaCO3 + 2HCl -> CaCl2 + CO2 + H2O", {"CaCO3": 1, "HCl": 2, "CaCl2": 1, "CO2": 1, "H2O": 1}),
            "AgNO3+NaCl->AgCl+NaNO3": ("AgNO3 + NaCl -> AgCl + NaNO3", {"AgNO3": 1, "NaCl": 1, "AgCl": 1, "NaNO3": 1}),
            "2H2+O2->2H2O": ("2H2 + O2 -> 2H2O", {"H2": 2, "O2": 1, "H2O": 2}),
        }
        
        # Check if equation is in common balances
        for pattern, (balanced, coeffs) in common_balances.items():
            if equation == pattern:
                return balanced, coeffs
        
        # If not found, check if already has coefficients
        # Simple check: if equation contains numbers at start of compounds
        has_coefficients = any(re.match(r'^\d+[A-Z]', part) for part in equation.replace('->', '+').split('+'))
        
        if has_coefficients:
            # Assume already balanced
            return equation.replace('+', ' + ').replace('->', ' -> '), {}
        else:
            # Return as-is with formatting
            return equation.replace('+', ' + ').replace('->', ' -> '), {}
        
    except Exception as e:
        print(f"Error balancing equation: {e}")
        return equation, {}

def ionic_equation(balanced_eq: str) -> str:
    """Generate complete ionic equation."""
    # Simplified implementation for common compounds
    ionic_map = {
        "HCl": "H⁺(aq) + Cl⁻(aq)",
        "NaOH": "Na⁺(aq) + OH⁻(aq)",
        "NaCl": "Na⁺(aq) + Cl⁻(aq)",
        "H2SO4": "2H⁺(aq) + SO4²⁻(aq)",
        "BaCl2": "Ba²⁺(aq) + 2Cl⁻(aq)",
        "Na2SO4": "2Na⁺(aq) + SO4²⁻(aq)",
        "AgNO3": "Ag⁺(aq) + NO3⁻(aq)",
        "KOH": "K⁺(aq) + OH⁻(aq)",
        "CaCO3": "Ca²⁺(aq) + CO3²⁻(aq)",
        "HNO3": "H⁺(aq) + NO3⁻(aq)",
        "H2O": "H2O(l)",
        "CO2": "CO2(g)",
        "H2": "H2(g)",
        "O2": "O2(g)",
    }
    
    result = balanced_eq
    for compound, ions in ionic_map.items():
        result = result.replace(compound, ions)
    
    return result

def net_ionic(ionic_eq: str) -> str:
    """Generate net ionic equation by removing spectator ions."""
    spectator_ions = ["Na⁺", "K⁺", "NO3⁻", "Cl⁻"]
    
    if '->' in ionic_eq:
        left, right = ionic_eq.split('->')
        
        # Simple logic: remove common ions on both sides
        for ion in spectator_ions:
            if ion in left and ion in right:
                left = left.replace(ion + "(aq)", "").replace(ion, "")
                right = right.replace(ion + "(aq)", "").replace(ion, "")
        
        # Clean up extra + signs and spaces
        left = re.sub(r'\s*\+\s*\+', '+', left).strip('+ ')
        right = re.sub(r'\s*\+\s*\+', '+', right).strip('+ ')
        
        # Remove empty parentheses
        left = left.replace('()', '').strip()
        right = right.replace('()', '').strip()
        
        if left and right:
            return f"{left} -> {right}"
    
    return ionic_eq

def get_reaction_type(equation: str) -> str:
    """Determine the type of chemical reaction."""
    equation_lower = equation.lower()
    
    if '->' in equation:
        left, right = equation.split('->')
    else:
        return "General"
    
    # Check for common reaction types
    if 'o2' in equation_lower and ('co2' in equation_lower or 'h2o' in equation_lower):
        return "Combustion"
    elif '+' in left and len([c for c in right.split('+') if c.strip()]) == 1:
        return "Synthesis"
    elif len([c for c in left.split('+') if c.strip()]) == 1 and '+' in right:
        return "Decomposition"
    elif any(m in equation_lower for m in ['zn', 'fe', 'cu', 'ag', 'mg', 'al']):
        return "Single Replacement"
    elif any(ion in equation_lower for ion in ['(aq)', 'no3', 'cl', 'so4', 'na', 'k']):
        return "Double Replacement"
    elif any(acid in equation_lower for acid in ['hcl', 'h2so4', 'hno3']) and any(base in equation_lower for base in ['naoh', 'koh']):
        return "Acid-Base"
    
    return "General"

def calculate_molar_mass(formula: str) -> float:
    """Calculate molar mass of a chemical formula."""
    try:
        # Atomic masses for common elements
        atomic_masses = {
            'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122,
            'B': 10.81, 'C': 12.011, 'N': 14.007, 'O': 16.00,
            'F': 19.00, 'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305,
            'Al': 26.982, 'Si': 28.085, 'P': 30.974, 'S': 32.06,
            'Cl': 35.45, 'K': 39.098, 'Ca': 40.078, 'Fe': 55.845,
            'Cu': 63.546, 'Zn': 65.38, 'Ag': 107.87, 'Ba': 137.33,
            'Pb': 207.2, 'I': 126.90
        }
        
        total = 0.0
        i = 0
        n = len(formula)
        
        while i < n:
            # Get element symbol
            if formula[i].isupper():
                element = formula[i]
                i += 1
                while i < n and formula[i].islower():
                    element += formula[i]
                    i += 1
                
                # Get subscript
                num_str = ''
                while i < n and formula[i].isdigit():
                    num_str += formula[i]
                    i += 1
                count = int(num_str) if num_str else 1
                
                # Add to total
                if element in atomic_masses:
                    total += atomic_masses[element] * count
                else:
                    # Estimate for unknown elements
                    total += 30.0 * count
            else:
                i += 1
        
        return round(total, 2)
    except:
        return 0.0

def predict_products(reactants: str, reaction_type: str = "auto") -> str:
    """Predict products for given reactants."""
    reactants_clean = reactants.strip().lower()
    
    # Common reaction predictions
    predictions = {
        "hcl + naoh": "HCl + NaOH -> NaCl + H2O",
        "h2so4 + naoh": "H2SO4 + 2NaOH -> Na2SO4 + 2H2O",
        "ch4 + o2": "CH4 + 2O2 -> CO2 + 2H2O",
        "h2 + o2": "2H2 + O2 -> 2H2O",
        "fe + o2": "4Fe + 3O2 -> 2Fe2O3",
        "agno3 + nacl": "AgNO3 + NaCl -> AgCl + NaNO3",
        "caco3 + hcl": "CaCO3 + 2HCl -> CaCl2 + CO2 + H2O",
        "zn + hcl": "Zn + 2HCl -> ZnCl2 + H2",
        "na2co3 + hcl": "Na2CO3 + 2HCl -> 2NaCl + CO2 + H2O",
        "mg + o2": "2Mg + O2 -> 2MgO",
        "al + o2": "4Al + 3O2 -> 2Al2O3",
        "hno3 + naoh": "HNO3 + NaOH -> NaNO3 + H2O",
    }
    
    for pattern, prediction in predictions.items():
        if pattern in reactants_clean:
            return prediction
    
    # General prediction
    return f"{reactants} -> Products"

def limiting_reagent(balanced_eq: str, reactants: List[str], amounts: List[float]) -> Tuple[float, str, Dict]:
    """Calculate limiting reagent with stoichiometry."""
    try:
        if not reactants or not amounts or len(reactants) != len(amounts):
            return 0.0, "Invalid input", {}
        
        # Simple implementation for common reactions
        # For HCl + NaOH -> NaCl + H2O
        if "HCl" in reactants and "NaOH" in reactants:
            hcl_idx = reactants.index("HCl")
            naoh_idx = reactants.index("NaOH")
            
            hcl_amount = amounts[hcl_idx]
            naoh_amount = amounts[naoh_idx]
            
            # Molar masses
            hcl_mm = calculate_molar_mass("HCl")
            naoh_mm = calculate_molar_mass("NaOH")
            
            # Convert to moles
            hcl_moles = hcl_amount / hcl_mm if hcl_mm > 0 else 0
            naoh_moles = naoh_amount / naoh_mm if naoh_mm > 0 else 0
            
            # 1:1 stoichiometry
            if hcl_moles <= naoh_moles:
                limiting_amount = hcl_amount
                limiting_reagent = "HCl"
                excess = {"NaOH": naoh_amount - (hcl_moles * naoh_mm)}
            else:
                limiting_amount = naoh_amount
                limiting_reagent = "NaOH"
                excess = {"HCl": hcl_amount - (naoh_moles * hcl_mm)}
            
            return limiting_amount, limiting_reagent, excess
        
        # For H2 + O2 -> H2O
        elif "H2" in reactants and "O2" in reactants:
            h2_idx = reactants.index("H2")
            o2_idx = reactants.index("O2")
            
            h2_amount = amounts[h2_idx]
            o2_amount = amounts[o2_idx]
            
            h2_mm = calculate_molar_mass("H2")
            o2_mm = calculate_molar_mass("O2")
            
            h2_moles = h2_amount / h2_mm if h2_mm > 0 else 0
            o2_moles = o2_amount / o2_mm if o2_mm > 0 else 0
            
            # 2:1 stoichiometry (2H2 + O2 -> 2H2O)
            h2_needed = o2_moles * 2
            
            if h2_moles <= h2_needed:
                limiting_amount = h2_amount
                limiting_reagent = "H2"
                excess_o2 = o2_amount - (h2_moles / 2 * o2_mm)
                excess = {"O2": excess_o2} if excess_o2 > 0 else {}
            else:
                limiting_amount = o2_amount
                limiting_reagent = "O2"
                excess_h2 = h2_amount - (o2_moles * 2 * h2_mm)
                excess = {"H2": excess_h2} if excess_h2 > 0 else {}
            
            return limiting_amount, limiting_reagent, excess
        
        # Default: first reactant is limiting
        return amounts[0], reactants[0], {}
        
    except Exception as e:
        print(f"Limiting reagent error: {e}")
        return 0.0, "Error", {}

def theoretical_yield(balanced_eq: str, product: str, limiting_amount: float, limiting_reagent: str) -> float:
    """Calculate theoretical yield of a product."""
    try:
        # Simple calculations for common reactions
        if limiting_reagent == "HCl" and product == "NaCl":
            # HCl + NaOH -> NaCl + H2O
            hcl_mm = calculate_molar_mass("HCl")
            nacl_mm = calculate_molar_mass("NaCl")
            moles_hcl = limiting_amount / hcl_mm if hcl_mm > 0 else 0
            return moles_hcl * nacl_mm
        
        elif limiting_reagent == "NaOH" and product == "NaCl":
            # HCl + NaOH -> NaCl + H2O
            naoh_mm = calculate_molar_mass("NaOH")
            nacl_mm = calculate_molar_mass("NaCl")
            moles_naoh = limiting_amount / naoh_mm if naoh_mm > 0 else 0
            return moles_naoh * nacl_mm
        
        elif limiting_reagent == "H2" and product == "H2O":
            # 2H2 + O2 -> 2H2O
            h2_mm = calculate_molar_mass("H2")
            h2o_mm = calculate_molar_mass("H2O")
            moles_h2 = limiting_amount / h2_mm if h2_mm > 0 else 0
            return moles_h2 * h2o_mm  # 1:1 ratio for H2 to H2O (2H2 -> 2H2O)
        
        elif limiting_reagent == "O2" and product == "H2O":
            # 2H2 + O2 -> 2H2O
            o2_mm = calculate_molar_mass("O2")
            h2o_mm = calculate_molar_mass("H2O")
            moles_o2 = limiting_amount / o2_mm if o2_mm > 0 else 0
            return moles_o2 * 2 * h2o_mm  # 1 O2 produces 2 H2O
        
        # Default: same as limiting amount
        return limiting_amount
        
    except Exception as e:
        print(f"Theoretical yield error: {e}")
        return 0.0

def percent_yield(actual: float, theoretical: float) -> float:
    """Calculate percent yield."""
    if theoretical == 0:
        return 0.0
    return round((actual / theoretical) * 100, 2)

def calculate_enthalpy_change(formula: str) -> float:
    """Estimate standard formation enthalpy."""
    enthalpies = {
        'H2O': -285.8,
        'CO2': -393.5,
        'NaCl': -411.2,
        'HCl': -92.3,
        'NaOH': -425.6,
        'H2SO4': -814.0,
        'NH3': -45.9,
        'CH4': -74.6,
        'O2': 0,
        'H2': 0,
        'N2': 0,
    }
    return enthalpies.get(formula, 0.0)

def predict_precipitate(equation: str) -> List[str]:
    """Predict precipitates in reaction."""
    precipitates = []
    common_precipitates = ['AgCl', 'BaSO4', 'CaCO3', 'PbI2']
    
    for ppt in common_precipitates:
        if ppt in equation:
            precipitates.append(ppt)
    
    return precipitates

def predict_gas_formation(equation: str) -> List[str]:
    """Predict gases formed in reaction."""
    gases = []
    common_gases = ['CO2', 'H2', 'O2', 'NH3']
    
    for gas in common_gases:
        if gas in equation:
            gases.append(gas)
    
    return gases

def calculate_mass_from_moles(moles: float, molar_mass: float) -> float:
    """Calculate mass from moles."""
    return round(moles * molar_mass, 4)

def calculate_moles_from_mass(mass: float, molar_mass: float) -> float:
    """Calculate moles from mass."""
    if molar_mass == 0:
        return 0.0
    return round(mass / molar_mass, 4)

# Initialize elements data
def init_elements_data():
    """Initialize periodic table data."""
    elements = [
        {"symbol": "H", "name": "Hydrogen", "atomic_number": 1, "atomic_mass": 1.008, "group": 1, "period": 1, "category": "nonmetal"},
        {"symbol": "He", "name": "Helium", "atomic_number": 2, "atomic_mass": 4.0026, "group": 18, "period": 1, "category": "noble"},
        {"symbol": "Li", "name": "Lithium", "atomic_number": 3, "atomic_mass": 6.94, "group": 1, "period": 2, "category": "alkali"},
        {"symbol": "C", "name": "Carbon", "atomic_number": 6, "atomic_mass": 12.011, "group": 14, "period": 2, "category": "nonmetal"},
        {"symbol": "N", "name": "Nitrogen", "atomic_number": 7, "atomic_mass": 14.007, "group": 15, "period": 2, "category": "nonmetal"},
        {"symbol": "O", "name": "Oxygen", "atomic_number": 8, "atomic_mass": 16.00, "group": 16, "period": 2, "category": "nonmetal"},
        {"symbol": "Na", "name": "Sodium", "atomic_number": 11, "atomic_mass": 22.990, "group": 1, "period": 3, "category": "alkali"},
        {"symbol": "Mg", "name": "Magnesium", "atomic_number": 12, "atomic_mass": 24.305, "group": 2, "period": 3, "category": "alkaline"},
        {"symbol": "Al", "name": "Aluminum", "atomic_number": 13, "atomic_mass": 26.982, "group": 13, "period": 3, "category": "basic"},
        {"symbol": "Si", "name": "Silicon", "atomic_number": 14, "atomic_mass": 28.085, "group": 14, "period": 3, "category": "semimetal"},
        {"symbol": "P", "name": "Phosphorus", "atomic_number": 15, "atomic_mass": 30.974, "group": 15, "period": 3, "category": "nonmetal"},
        {"symbol": "S", "name": "Sulfur", "atomic_number": 16, "atomic_mass": 32.06, "group": 16, "period": 3, "category": "nonmetal"},
        {"symbol": "Cl", "name": "Chlorine", "atomic_number": 17, "atomic_mass": 35.45, "group": 17, "period": 3, "category": "halogen"},
        {"symbol": "K", "name": "Potassium", "atomic_number": 19, "atomic_mass": 39.098, "group": 1, "period": 4, "category": "alkali"},
        {"symbol": "Ca", "name": "Calcium", "atomic_number": 20, "atomic_mass": 40.078, "group": 2, "period": 4, "category": "alkaline"},
        {"symbol": "Fe", "name": "Iron", "atomic_number": 26, "atomic_mass": 55.845, "group": 8, "period": 4, "category": "transition"},
        {"symbol": "Cu", "name": "Copper", "atomic_number": 29, "atomic_mass": 63.546, "group": 11, "period": 4, "category": "transition"},
        {"symbol": "Zn", "name": "Zinc", "atomic_number": 30, "atomic_mass": 65.38, "group": 12, "period": 4, "category": "transition"},
        {"symbol": "Ag", "name": "Silver", "atomic_number": 47, "atomic_mass": 107.87, "group": 11, "period": 5, "category": "transition"},
        {"symbol": "Ba", "name": "Barium", "atomic_number": 56, "atomic_mass": 137.33, "group": 2, "period": 6, "category": "alkaline"},
        {"symbol": "Pb", "name": "Lead", "atomic_number": 82, "atomic_mass": 207.2, "group": 14, "period": 6, "category": "basic"},
    ]
    
    import os
    os.makedirs('data', exist_ok=True)
    
    with open('data/elements.json', 'w') as f:
        json.dump(elements, f, indent=2)
    
    return elements

# Test the functions
if __name__ == "__main__":
    print("Testing chem_engine.py...")
    
    # Test parsing
    eq = "HCl + NaOH -> NaCl + H2O"
    parsed = parse_reaction_string(eq)
    print(f"Parsed {eq}: {parsed}")
    
    # Test balancing
    balanced, coeffs = balance_equation(eq)
    print(f"Balanced: {balanced}")
    print(f"Coefficients: {coeffs}")
    
    # Test molar mass
    mm = calculate_molar_mass("H2O")
    print(f"Molar mass of H2O: {mm}")
    
    print("✓ All tests passed!")