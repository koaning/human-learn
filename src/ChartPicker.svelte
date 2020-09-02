<script>
    import { fade, fly } from 'svelte/transition';
    import { createEventDispatcher } from 'svelte';

    const dispatch = createEventDispatcher();

    export let chartInfo;
</script>

<div in:fly="{{ duration: 1000 }}" out:fade>
    <button>
        Basic Tree
    </button>
    <button>
        Histogram Selector
    </button>
    <button on:click={() => dispatch('delete')}>
        x
    </button>
    <p>
        {JSON.stringify(chartInfo)}
    </p>
</div>
