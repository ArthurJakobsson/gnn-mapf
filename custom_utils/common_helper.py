import os

def str2bool(v: str) -> bool:
    """Converts a string to a boolean value. Used for argparse."""
    return v.lower() in ("yes", "true", "t", "1")


# ### Managing file names
# def getMapBDScenAgents(filepath):
#     """Input: filepath to a scen or txt file
#     Output: mapname, whichbd, custom_scenname, num agents
#     Examples: 
#     - [WHICHBD].[CUSTOMSCENNAME].[NUMAGENTS].txt 
#     - [WHICHBD].[CUSTOMSCENNAME].[NUMAGENTS].scen 
#     - [WHICHBD].scen, these are the default scens files 
#     """
#     filename = os.path.basename(filepath) # This gets the filename from the path

#     splits = filename.split('.')
#     assert(len(splits) == 2 or len(splits) == 4)

#     whichbd = splits[0]
#     mapname = whichbd.split('-')[0]

#     if len(splits) == 2:
#         custom_scen = splits[0]
#         num_agents = 0
#     else:
#         custom_scen = splits[1]
#         # Get num_agents
#         num_agents = int(splits[2])

#     return mapname, whichbd, custom_scen, num_agents


### Managing file names
def getMapScenAgents(filepath):
    """Input: filepath to a scen or txt file
    Output: mapname, custom_scenname, num agents
    Examples: 
    - [CUSTOMSCENNAME].[NUMAGENTS].txt 
    - [CUSTOMSCENNAME].[NUMAGENTS].scen 
    - [CUSTOMSCENNAME].scen, these are the default scens files 
    """
    filename = os.path.basename(filepath) # This gets the filename from the path

    splits = filename.split('.')
    assert(len(splits) == 2 or len(splits) == 3)

    custom_scenname = splits[0]
    mapname = custom_scenname.split('-')[0]

    if len(splits) == 2:
        num_agents = 0
    else:
        num_agents = int(splits[1])

    return mapname, custom_scenname, num_agents