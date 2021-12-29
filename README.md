# Crysto

This module provides the ability to generate crystal structures, either random or predefined and then simulate the respecitve X-ray powder diffraction pattern with configurable peak shape and instrument parameters.

# Random Structure

```
cell = Struct().create(Cubic())
pattern = PatternGen(cell)
pattern = pattern.calculate_powder_pattern(180.0,0.01,0.1,-0.0001,0.0001,0.001,0.5,0.001,8.0,1.0)
```

# Defined Structure

```
at = [['62', '0.0', '0.0', '0.0', '1.0', '1.0']]
cell = Struct(random=False).create(Tetragonal(), a=6, c=8, sg=78, atomTable=at)
pattern = Diffraction(cell)
pattern = pattern.calculate_powder_pattern(180.0, 0.01, 0.1, -0.0001, 00001,0.001, 0.5, 0.001, 8.0, 1.0)
```
