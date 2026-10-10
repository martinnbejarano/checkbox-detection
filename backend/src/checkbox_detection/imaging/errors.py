"""Why an upload was rejected. No HTTP knowledge: `api/errors.py` maps these to status codes."""


class ImagingError(Exception):
    """Base for every rejected upload."""


class UnsupportedFormat(ImagingError):
    """The bytes are not one of the accepted formats."""


class InputTooLarge(ImagingError):
    """The decoded image would exceed a configured size cap."""


class CorruptInput(ImagingError):
    """The format was recognized but the file could not be decoded."""
