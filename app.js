// set the dimensions and margins of the graph
var margin = {top: 10, right: 30, bottom: 30, left: 60},
	width = 460 - margin.left - margin.right,
	height = 400 - margin.top - margin.bottom;

// append the svg object to the body of the page
var svg = d3.select("#my_dataviz")
  .append("svg")
	.attr("width", width + margin.left + margin.right)
	.attr("height", height + margin.top + margin.bottom)
  .append("g")
	.attr("transform",
		  "translate(" + margin.left + "," + margin.top + ")");

//Read the data
all_data = 0;
d3.csv("peng.csv", function(data) {
  var col1 = "bill_length_mm",
	  col2 = "bill_depth_mm";
  var label = "species";
  var uniq_labels = data.map(d => d["species"]).filter(only_unique)
  for(i in uniq_labels){
	d3.select("select#label")
	  .append("option")
	  .attr("value", uniq_labels[i])
	  .text(uniq_labels[i])
  }

  var color = d3.scaleOrdinal()
	.domain(uniq_labels)
	.range(["#440154ff", "#21908dff", "#fde725ff"])

  all_data = data;

  // Add X axis
  var x = d3.scaleLinear()
	.domain(calc_norm_extend(data, col1))
	.range([ 0, width ]);

  svg.append("g")
	.attr("transform", "translate(0," + height + ")")
	.call(d3.axisBottom(x));

  // Add Y axis
  var y = d3.scaleLinear()
	.domain(calc_norm_extend(data, col2))
	.range([ height, 0]);
  svg.append("g")
	.call(d3.axisLeft(y));

  // Add dots
  svg.append('g')
	.attr("class", "point-group")
	.selectAll("dot")
	.data(data)
	.enter()
	.append("circle")
	  .attr("cx", function (d) { return x(d[col1]); } )
	  .attr("cy", function (d) { return y(d[col2]); } )
	  .style("fill", function(d){ return color(d[label])})
	  .attr("r", 3.5)

var group_stats = d3.select("svg").node().getBBox();
var draw_group = svg
	.append("rect")
	.attr("x", 0)
	.attr("y", 0)
	.attr("width", group_stats["width"])
	.attr("height", group_stats["height"])
	.attr("opacity", 0.0)
	.attr("class", "draw-group");

	var dragging = false, drawing = false, startPoint;
	var points = [], g;
	// behaviors
	var dragger = d3.drag()
		.on('drag', handleDrag)
		.on('end', function(d){
			dragging = false;
		});

	svg.on('mouseup', function(){
		if(dragging) return;
		drawing = true;
		startPoint = [d3.mouse(this)[0], d3.mouse(this)[1]];
		if(svg.select('g.drawPoly').empty()) g = svg.append('g').attr('class', 'drawPoly');
		if(d3.event.target.hasAttribute('is-handle')) {
			closePolygon();
			return;
		};
		points.push(d3.mouse(this));
		g.select('polyline').remove();
		var polyline = g.append('polyline').attr('points', points)
						.style('fill', 'none')
						.attr('stroke', '#000');
		for(var i = 0; i < points.length; i++) {
			g.append('circle')
			.attr("class", "drawn")
			.attr('cx', points[i][0])
			.attr('cy', points[i][1])
			.attr('r', 4)
			.attr('fill', 'darkgray')
			.attr('stroke', '#000')
			.attr('is-handle', 'true')
			.style({cursor: 'pointer'});
		}
	});

	function closePolygon() {
		svg.select('g.drawPoly').remove();
		var g = svg.append('g');
		var selected_class = d3.select("select").node().value;
		g.append('polygon')
		.attr('points', points)
		.style("opacity", 0.3)
		.attr("label", selected_class)
		.style('fill', color(selected_class));
		drawing = false;
	}

	svg.on('mousemove', function() {
		if(!drawing) return;
		var g = d3.select('g.drawPoly');
		g.select('line').remove();
		var line = g.append('line')
					.attr('x1', startPoint[0])
					.attr('y1', startPoint[1])
					.attr('x2', d3.mouse(this)[0] + 2)
					.attr('y2', d3.mouse(this)[1])
					.attr('stroke', '#53DBF3')
					.attr('stroke-width', 1);
	})

	function handleDrag() {
		if(drawing) return;
		var dragCircle = d3.select(this), newPoints = [], circle;
		dragging = true;
		var poly = d3.select(this.parentNode).select('polygon');
		var circles = d3.select(this.parentNode).selectAll('circle.drawn');
		console.log(circles)
		dragCircle
		  .attr('cx', d3.event.x)
		  .attr('cy', d3.event.y);
		for (var i = 0; i < circles[0].length; i++) {
			circle = d3.select(circles[0][i]);
			newPoints.push([circle.attr('cx'), circle.attr('cy')]);
		}
		poly.attr('points', newPoints);
	}

})
