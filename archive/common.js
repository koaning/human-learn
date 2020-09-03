function only_unique(value, index, self) { 
    return self.indexOf(value) === index;
}

function calc_norm_extend(data, colname, padding=0.15){
  var extent = d3.extent(data.map(d => d[colname])).map(Number);
  extent[0] = extent[0] - padding * (d3.max(extent) - d3.min(extent))
  extent[1] = extent[1] + padding * (d3.max(extent) - d3.min(extent))
  return extent
}