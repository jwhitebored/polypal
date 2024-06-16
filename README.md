# polypal
Neural net made in TensorFlow to determine the degree of a polynomial given a noisy, polynomial signal. Intended use is to use the degree in standard curve fitting algorithms (non-linear regression) like SciPy's opt.curve_fit() or NumPy's np.polyfit() which require a guess at the type of function your data represents.
