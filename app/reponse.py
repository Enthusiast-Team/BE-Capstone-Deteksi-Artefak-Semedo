from flask import jsonify, make_response



def success(values, massage):
    res = {
        'data' : values,
        'massage' : massage 
    }
    return make_response(jsonify(res)), 200


def badRequest(values, massage):
    res = {
        'data': values,
        'maasage': massage
    }
    return make_response(jsonify(res)),400