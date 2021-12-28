class Struct:
    """A class for creating structures."""

    def __init__(self, random=True):
        self._random = random

    def create(self, cell, sg=None, atomTable=None, **kwargs):
        cell.addCellParams(random=self._random, **kwargs)
        cell.addSpaceGroup(random=self._random, sg=sg)
        cell.addAtoms(random=self._random, atomTable=atomTable)
        cell.buildUnitCell()
        return cell
