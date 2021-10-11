# Crysto

Generate random crystal structures and respecitve X-ray diffraction patterns.


# Random Cell

cell = CrystoGen()
pattern = PatternGen(cell)
pattern = pattern.calculatePowderPattern(180.0,0.01,0.1,-0.0001,0.0001,0.001,0.5,0.001,8.0,1.0)


# Specified Cell

at = [['181', '0.0', '0.0', '0.0', '1.0','1.0'],['181', '-0.5', '-0.5', '0.0', '1.0','1.0'],['116','0.0','0.0','0.25','1.0','1.0'],['116','-0.209','-0.295','0.0','1.0','1.0'],['10','0.073','0.474','0.218','1.0','1.0'],['10','0.073','0.474','0.218','1.0','1.0']]
#cell = CrystoGen(SpaceGroup=1, Info=True)
cell = CrystoGen(atomTable=at,info=True)
pattern = PatternGen(cell)
pattern = pattern.calculatePowderPattern(180.0,0.01,0.1,-0.0001,0.0001,0.001,0.5,0.001,8.0,1.0)


# Batch Data Generate
genrateTrainingData(1,"")