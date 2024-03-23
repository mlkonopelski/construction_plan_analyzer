from fastapi import Depends, HTTPException, Header
from utils.config import Settings, get_settings


def verify_auth(authorization = Header(None), settings: Settings = Depends(get_settings)):
    '''
    Authorization: Bearer <token>
    {'authorization': 'Beaer <token>'}
    '''
    if settings.debug and settings.skip_auth:
        return 
    if authorization is None:
        raise HTTPException(detail='No access', status_code=401)
    label, token = authorization.split()
    if token != settings.app_auth_token:
        raise HTTPException(detail='No access', status_code=401)