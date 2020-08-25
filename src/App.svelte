<script>
	import AddChartButton from './AddChartButton.svelte';
	import LabelSelector from './LabelSelector.svelte';

	let fp = "filepath";
	let labelname = "label";
	let labelvals = [];

    fetch(window.location.href + "info").then(d => d.json()).then(d => labelname = d['label']);
	fetch(window.location.href + "info").then(d => d.json()).then(d => fp = d['filepath']);
	fetch(window.location.href + "uniq_labels").then(d => d.json()).then(d => labelvals = d);

	let charts = [];
</script>

<main>
	<h1>Human Learn - Filepath: <span><code>{fp}</code></span></h1>
	<p>This tool will allow you to create a model using natural intelligence.</p>
	<LabelSelector labelname={labelname} labelvals={labelvals}></LabelSelector>
	<AddChartButton charts={charts}></AddChartButton>
</main>

<style>

</style>
