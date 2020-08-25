<script>
	import AddChartButton from './AddChartButton.svelte';
	let fp;
	fetch(window.location.href + "info").then(d => d.json()).then(d => fp = d['filepath']);
</script>

<main>
	<h1>Human Learn - Filepath: <span><code>{fp}</code></span></h1>
	<p>This tool will allow you to create a model using natural intelligence.</p>
	<AddChartButton></AddChartButton>
</main>

<style>

</style>
