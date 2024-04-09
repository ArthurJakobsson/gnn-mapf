import os


def main():
  map_path = "data/mapf-map"
  scen_path = "data/scen-random"

  #fix inside scenes map name
  for filename in os.listdir(scen_path):
    f = os.path.join(scen_path, filename)
    if f[-9:] == ".DS_Store": continue
    if os.path.isfile(f):
      my_file = open(f, "r+")
      newlines = list()
      for line in my_file.readlines():
        new_name = list(line)
        for i, c in enumerate(new_name):
          if (c == '-'):
              new_name[i] = '_'
        newlines.append(''.join(new_name))
      str_to_write = ''.join(newlines)
      my_file.close()
      os.remove(f)
      with open(f, "w") as write_file:
         write_file.write(str_to_write)
    else:
        raise RuntimeError("bad path dir")


  #rename scenes
  for filename in os.listdir(scen_path):
      f = os.path.join(scen_path, filename)
      if os.path.isfile(f):
        count_dashcula = filename.count('-')
        if(count_dashcula>2):
          new_name = list(filename)
          replace_count = 0
          for i, c in enumerate(new_name):
             if (replace_count == (count_dashcula-2)):
                break
             if (c == '-'):
                replace_count+=1
                new_name[i] = '_'
          os.rename(f, os.path.join(scen_path, ''.join(new_name)))
      else:
          raise RuntimeError("bad path dir")

  #rename maps
  for filename in os.listdir(map_path):
      f = os.path.join(map_path, filename)
      if os.path.isfile(f):
        count_dashcula = filename.count('-')
        if(count_dashcula>0):
          new_name = list(filename)
          for i, c in enumerate(new_name):
             if (c == '-'):
                new_name[i] = '_'
          os.rename(f, os.path.join(map_path, ''.join(new_name)))
      else:
          raise RuntimeError("bad path dir")




  pass



if __name__ == "__main__":
    main()