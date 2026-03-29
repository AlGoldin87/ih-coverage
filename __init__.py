from .ih_coverage import check_coverage, discretize, _suggest_sharpness

def suggest_sharpness(data, alpha=1.0):
    """
    Suggest optimal sharpness for a single column.
    
    Parameters:
    - data: 1D numpy array (float32)
    - alpha: penalty coefficient for interval unreliability (default 1.0)
    
    Returns:
    - optimal sharpness value
    """
    return _suggest_sharpness(data, alpha)

__all__ = ['check_coverage', 'discretize', 'suggest_sharpness']
