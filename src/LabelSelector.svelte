<script>
    export let labelname;
    export let labelvals;
</script>

<p>
    <span>Labels: <code>{labelname}</code></span>
    <span>: </span>
    {#each labelvals as v}
        <code>{v} </code>
    {/each}
 </p>
