# This is a list of errors the Webservice returns. You can come up
# with new error codes and plug them into the API.
#
# An example JSON response for an error looks like this:
#
#   { "status" : failed, "message" : "IMAGE_DECODE_ERROR", "code" : 10 }
#
# If there are multiple errors, only the first error is considered.

IMAGE_DECODE_ERROR = 10
IMAGE_RESIZE_ERROR = 11
PREDICTION_ERROR = 12
SERVICE_TEMPORARY_UNAVAILABLE = 20
UNKNOWN_ERROR = 21
INVALID_FORMAT = 30
INVALID_API_KEY = 31
INVALID_API_TOKEN = 32
MISSING_ARGUMENTS = 40

errors = {
    IMAGE_DECODE_ERROR : "IMAGE_DECODE_ERROR",
    IMAGE_RESIZE_ERROR  : "IMAGE_RESIZE_ERROR",
    SERVICE_TEMPORARY_UNAVAILABLE    : "SERVICE_TEMPORARILY_UNAVAILABLE",
    PREDICTION_ERROR : "PREDICTION_ERROR",
    UNKNOWN_ERROR : "UNKNOWN_ERROR",
    INVALID_FORMAT : "INVALID_FORMAT",
    INVALID_API_KEY : "INVALID_API_KEY",
    INVALID_API_TOKEN : "INVALID_API_TOKEN",
    MISSING_ARGUMENTS : "MISSING_ARGUMENTS"
}

# The WebAppException might be useful. It enables us to 
# throw exceptions at any place in the application and give the user
# a custom error code.
class WebAppException(Exception):

    def __init__(self, error_code, status_code=None):
        Exception.__init__(self)
        self.status_code = 400
        self.error_code = error_code
        try:
            self.message = errors[self.error_code]
        except:
            self.error_code = UNKNOWN_ERROR
            self.message = errors[self.error_code]
        if status_code is not None:
            self.status_code = status_code

    def to_dict(self):
        rv = dict()
        rv['status'] = 'failed'
        rv['code'] = self.error_code
        rv['message'] = self.message
        return rv

# Wow, a decorator! This enables us to catch Exceptions 
# in a method and raise a new WebAppException with the 
# original Exception included. This is a quick and dirty way
# to minimize error handling code in our server.
class ThrowsWebAppException(object):
    def __init__(self, error_code, status_code=None):
        self.error_code = error_code
        self.status_code = status_code

    def __call__(self, function):
        def returnfunction(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as e:
                raise WebAppException(self.error_code, e)
        return returnfunction
