# Pyfit
### This an ongoing project!

```
pip install ifitpy
```
# Fitter Package Instructions  

## 1. Creating a Fitter Instance  
```python
f = Fitter("linear")  # Available types: "linear", "expo", "gaussian", "gaussian2d", "poly"
```

This package provides fitting capabilities for given `x, y` data, encapsulating both `iminuit` and `curve_fit`.  

Functions are categorized as:  
- **Simple**: `"linear"`, `"expo"`  
- **Complex**: `"gaussian"`, `"gaussian2d"`, `"poly"`

## 2. Performing a Fit  

### Simple Functions (`linear`, `expo`)
```python
f.fit(x, y)  # Estimates initial parameters automatically  
f.fit(x, y, p0)  # Uses provided initial parameters (p0)
```

### Complex Functions (`gaussian`, `gaussian2d`, `poly`)
```python
f.fit(x, y, n)  # Fits using `n` components (e.g., a sum of `n` Gaussians or an `n`-degree polynomial)  
f.fit(x, y, p0)  # Uses provided initial parameters  
f.fit(x, y, n, p0)  # Uses both `n` components and provided initialization parameters  
```
> **Note:** For `fit(x, y, n, p0)`, the length of `p0` must be `n * parameters_to_fit`.

## 3. Binned Fitting  
The `fitBinned` function allows fitting a profile histogram instead of raw data, which is often faster and accounts for statistical fluctuations.  
```python
f.fitBinned(x, y, bins=50)
```

## 4. Extracting Fit Results  
```python
f.fit([0, 10], [0, -10])
p = f.getParams()

print(p)         # Prints a summary of available variables  
print(p.vars)    # List of fit results  
print(p.m)       # Slope for the "linear" type  
print(p.b)       # Intercept for the "linear" type  
```

## 5. Evaluating the Fitted Function  
```python
print(f.evaluate([20]))  # Evaluates the function at x = 20 (useful for plotting)
```


